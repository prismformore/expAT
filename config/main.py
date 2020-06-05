import settings
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=settings.device_id
import sys
import argparse
import csv
import numpy as np
import time
import torch
from torch import nn
import  torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from criterions import expATLoss, CrossEntropyLabelSmoothLoss
import torchvision.transforms as transforms

logger = settings.logger
torch.cuda.manual_seed_all(66)
torch.manual_seed(66)

from datasets import SYSU_triplet_dataset, SYSU_eval_datasets, Image_dataset
import itertools
import solver
from models import IdClassifier, FeatureEmbedder, Baseline
from eval import test, evaluate


def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

class Session:
    def __init__(self):
        self.log_dir = settings.log_dir
        self.model_dir = settings.model_dir
        ensure_dir(settings.log_dir)
        ensure_dir(settings.model_dir)
        logger.info('set log dir as %s' % settings.log_dir)
        logger.info('set model dir as %s' % settings.model_dir)

        ##################################### Import models ###########################
        self.feature_generator = Baseline(last_stride=1, model_path='/app/pretrained_models/resnet50-19c8e357.pth')

        self.feature_embedder_rgb = FeatureEmbedder(2048)
        self.feature_embedder_ir = FeatureEmbedder(2048)
        self.id_classifier = IdClassifier()

        if torch.cuda.is_available():
            self.feature_generator.cuda()
            self.feature_embedder_rgb.cuda()
            self.feature_embedder_ir.cuda()
            self.id_classifier.cuda()

        self.feature_generator = nn.DataParallel(self.feature_generator, device_ids=range(settings.num_gpu))

        self.feature_embedder_rgb = nn.DataParallel(self.feature_embedder_rgb, device_ids=range(settings.num_gpu))
        self.feature_embedder_ir = nn.DataParallel(self.feature_embedder_ir, device_ids=range(settings.num_gpu))
        self.id_classifier = nn.DataParallel(self.id_classifier, device_ids=range(settings.num_gpu))

        ############################# Get Losses & Optimizers #########################
        self.criterion_at = expATLoss()
        self.criterion_identity = CrossEntropyLabelSmoothLoss(settings.num_classes, epsilon=0.1) #torch.nn.CrossEntropyLoss()

        opt_models = [self.feature_generator,
                      self.feature_embedder_rgb,
                      self.feature_embedder_ir,
                      self.id_classifier]

        def make_optimizer(opt_models):
            train_params = []

            for opt_model in opt_models:
                for key, value in opt_model.named_parameters():
                    if not value.requires_grad:
                        continue
                    lr = settings.BASE_LR
                    weight_decay = settings.WEIGHT_DECAY
                    if "bias" in key:
                        lr = settings.BASE_LR * settings.BIAS_LR_FACTOR
                        weight_decay = settings.WEIGHT_DECAY_BIAS
                    train_params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]


            optimizer = torch.optim.Adam(train_params)
            return optimizer

        self.optimizer_G = make_optimizer(opt_models)

        self.epoch_count = 0
        self.step = 0
        self.save_steps = settings.save_steps
        self.num_workers = settings.num_workers
        self.writers = {}
        self.dataloaders = {}

        self.sche_G = solver.WarmupMultiStepLR(self.optimizer_G, milestones=settings.iter_sche, gamma=0.1) # default setting 

    def tensorboard(self, name):
        self.writers[name] = SummaryWriter(os.path.join(self.log_dir, name + '.events'))
        return self.writers[name]


    def write(self, name, out):
        for k, v in out.items():
            self.writers[name].add_scalar(name + '/' + k, v, self.step)


        out['G_lr'] = self.optimizer_G.param_groups[0]['lr']
        out['step'] = self.step
        out['eooch_count'] = self.epoch_count
        outputs = [
            "{}:{:.4g}".format(k, v) 
            for k, v in out.items()
        ]
        logger.info(name + '--' + ' '.join(outputs))

    def save_checkpoints(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        obj = {
            'feature_generator': self.feature_generator.state_dict(),
            'feature_embedder_rgb': self.feature_embedder_rgb.state_dict(),
            'feature_embedder_ir': self.feature_embedder_ir.state_dict(),
            'id_classifier': self.id_classifier.state_dict(),
            'clock': self.step,
            'epoch_count': self.epoch_count,
            'opt_G': self.optimizer_G.state_dict(),
        }
        torch.save(obj, ckp_path)

    def load_checkpoints(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        try:
            obj = torch.load(ckp_path)
            print('load checkpoint: %s' %ckp_path)
        except FileNotFoundError:
            return
        self.feature_generator.load_state_dict(obj['feature_generator'])
        self.feature_embedder_rgb.load_state_dict(obj['feature_embedder_rgb'])
        self.feature_embedder_ir.load_state_dict(obj['feature_embedder_ir'])
        self.id_classifier.load_state_dict(obj['id_classifier'])
        self.optimizer_G.load_state_dict(obj['opt_G'])
        self.step = obj['clock']
        self.epoch_count = obj['epoch_count']
        self.sche_G.last_epoch = self.step


    def load_checkpoints_delf_init(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        obj = torch.load(ckp_path)
        self.backbone.load_state_dict(obj['backbone'])

    def cal_fea(self, x, domain_mode):
        feat = self.feature_generator(x)
        if domain_mode == 'rgb':
            return self.feature_embedder_rgb(feat)
        elif domain_mode == 'ir':
            return self.feature_embedder_ir(feat)


    def inf_batch(self, batch):
        alpha = settings.alpha
        beta = settings.beta

        anchor_rgb, positive_rgb, negative_rgb, anchor_ir, positive_ir, \
        negative_ir, anchor_label, modality_rgb, modality_ir = batch

        if torch.cuda.is_available():
            anchor_rgb = anchor_rgb.cuda()
            positive_rgb = positive_rgb.cuda()
            negative_rgb = negative_rgb.cuda()
            anchor_ir = anchor_ir.cuda()
            positive_ir = positive_ir.cuda()
            negative_ir = negative_ir.cuda()
            anchor_label = anchor_label.cuda()


        anchor_rgb_features = self.cal_fea(anchor_rgb, 'rgb')
        positive_rgb_features = self.cal_fea(positive_rgb, 'rgb')
        negative_rgb_features = self.cal_fea(negative_rgb, 'rgb')

        anchor_ir_features = self.cal_fea(anchor_ir, 'ir')
        positive_ir_features = self.cal_fea(positive_ir, 'ir')
        negative_ir_features = self.cal_fea(negative_ir, 'ir')


        at_loss_rgb = self.criterion_at.forward(anchor_rgb_features, 
            positive_ir_features, negative_ir_features)

        at_loss_ir = self.criterion_at.forward(anchor_ir_features, 
            positive_rgb_features, negative_rgb_features)


        at_loss = at_loss_rgb + at_loss_ir

        predicted_id_rgb = self.id_classifier(anchor_rgb_features)
        predicted_id_ir = self.id_classifier(anchor_ir_features)

        identity_loss = self.criterion_identity(predicted_id_rgb, anchor_label) + \
                        self.criterion_identity(predicted_id_ir, anchor_label)

        loss_G = alpha*at_loss + beta*identity_loss

        self.optimizer_G.zero_grad() 
        loss_G.backward()
        self.optimizer_G.step()

        self.write('train_stats', {'loss_G': loss_G, 
                                   'at_loss': at_loss, 
                                   'identity_loss': identity_loss
        })


def run_train_val(ckp_name='ckp_latest'):
    sess = Session()
    sess.load_checkpoints(ckp_name)

    sess.tensorboard('train_stats')
    sess.tensorboard('val_stats')

    ######################## Get Datasets & Dataloaders ###########################

    train_dataset = SYSU_triplet_dataset(data_folder=settings.data_folder, transforms_list=settings.transforms_list)

    def get_train_dataloader():
        return iter(DataLoader(SYSU_triplet_dataset(data_folder=settings.data_folder, transforms_list=settings.transforms_list), batch_size=settings.train_batch_size, shuffle=True,num_workers=settings.num_workers, drop_last = True))

    train_dataloader = get_train_dataloader()

    eval_val = SYSU_eval_datasets(data_folder=settings.data_folder, data_split='val')

    transform_test = settings.test_transforms_list

    val_queryloader = DataLoader(
        Image_dataset(eval_val.query, transform=transform_test),
        batch_size=settings.val_batch_size, shuffle=False, num_workers=0,
        drop_last=False,
    )

    val_galleryloader = DataLoader(
        Image_dataset(eval_val.gallery, transform=transform_test),
        batch_size=settings.val_batch_size, shuffle=False, num_workers=0,
        drop_last=False,
    )

    while sess.step < settings.iter_sche[-1]:
        sess.sche_G.step()
        sess.feature_generator.train()
        sess.feature_embedder_rgb.train()
        sess.feature_embedder_ir.train()

        sess.id_classifier.train()

        try:
            batch_t = next(train_dataloader)
        except StopIteration:
            train_dataloader = get_train_dataloader()
            batch_t = next(train_dataloader)
            sess.epoch_count += 1

        sess.inf_batch(batch_t)

        if sess.step % int(settings.latest_steps) == 0:
            sess.save_checkpoints('ckp_latest')
            sess.save_checkpoints('ckp_latest_backup')

        if sess.step % settings.val_step ==0:
            sess.feature_generator.eval()
            sess.feature_embedder_rgb.eval()
            sess.feature_embedder_ir.eval()
            sess.id_classifier.eval()
            test_ranks, test_mAP = test([nn.Sequential(sess.feature_generator, sess.feature_embedder_rgb), nn.Sequential(sess.feature_generator, sess.feature_embedder_ir)], val_queryloader, val_galleryloader)

            sess.write('val_stats', {'test_mAP_percentage': test_mAP*100.0, \
                                     'test_rank-1_accuracy_percentage':test_ranks[0]*100.0,\
                                     'test_rank-5_accuracy_percentage':test_ranks[4]*100.0,\
                                     'test_rank-10_accuracy_percentage':test_ranks[9]*100.0,\
                                     'test_rank-20_accuracy_percentage':test_ranks[19]*100.0
            })

        if sess.step % sess.save_steps == 0:
            sess.save_checkpoints('ckp_step_%d' % sess.step)
            logger.info('save model as ckp_step_%d' % sess.step)
        sess.step += 1


def run_test(ckp, setting):
    if ckp == 'all':
        models = sorted(os.listdir('../models/'))
        csvfile = open('all_test_results.csv', 'w')
        writer = csv.writer(csvfile)

        writer.writerow(['ckp_name', 'mAP', 'R1', 'R5', 'R10', 'R20'])

        for mm in models:
            result = test_ckp(mm, setting)
            writer.writerow(result)

        csvfile.close()

    else:
        test_ckp(ckp, setting)


def test_ckp(ckp_name, setting):
    sess = Session()
    sess.load_checkpoints(ckp_name)

    search_mode = setting.split('_')[0] # 'all' or 'indoor'
    search_setting = setting.split('_')[1] # 'single' or 'multi'

    transform_test = settings.test_transforms_list

    results_ranks = np.zeros(50)
    results_map = np.zeros(1)
    
    for i in range(settings.test_times):
        eval_test = SYSU_eval_datasets(data_folder=settings.data_folder, data_split='test', search_mode=search_mode, search_setting=search_setting, use_random=True)

        test_queryloader = DataLoader(
            Image_dataset(eval_test.query, transform=transform_test),
            batch_size=settings.val_batch_size, shuffle=False, num_workers=0,
            drop_last=False,
        )

        test_galleryloader = DataLoader(
            Image_dataset(eval_test.gallery, transform=transform_test),
            batch_size=settings.val_batch_size, shuffle=False, num_workers=0,
            drop_last=False,
        )

        test_ranks, test_mAP = test([nn.Sequential(sess.feature_generator, sess.feature_embedder_rgb), nn.Sequential(sess.feature_generator, sess.feature_embedder_ir)], test_queryloader, test_galleryloader)
        results_ranks += test_ranks
        results_map += test_mAP

        logger.info('Test no.{} for model {} in setting {}, Test mAP: {}, R1: {}, R5: {}, R10: {}, R20: {}'.format(i,
                                                                                                                   ckp_name,
                                                                                                                   setting,
                                                                                                                   test_mAP*100.0,
                                                                                                                   test_ranks[0]*100.0,
                                                                                                                   test_ranks[4]*100.0,
                                                                                                                   test_ranks[9]*100.0,
                                                                                                                   test_ranks[19]*100.0))
        

    test_mAP = results_map / settings.test_times
    test_ranks = results_ranks / settings.test_times
    logger.info('For model {} in setting {}, AVG test mAP: {}, R1: {}, R5: {}, R10: {}, R20: {}'.format(ckp_name,
                                                                                                        setting,
                                                                                                        test_mAP*100.0,
                                                                                                        test_ranks[0]*100.0,
                                                                                                        test_ranks[4]*100.0,
                                                                                                        test_ranks[9]*100.0,
                                                                                                        test_ranks[19]*100.0))

    return [ckp_name, test_mAP*100.0, test_ranks[0]*100.0, test_ranks[4]*100.0, test_ranks[9]*100.0, test_ranks[19]*100.0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--action', default='train')
    parser.add_argument('-m', '--model', default='ckp_latest')
    parser.add_argument('-s', '--setting', default='all_single')

    args = parser.parse_args(sys.argv[1:])

    if args.action == 'train':
        run_train_val(args.model)
    elif args.action == 'test':
        run_test(args.model, args.setting)

