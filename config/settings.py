import os
import logging
import transforms

G_lr = 3e-4
BASE_LR = 3e-4
BIAS_LR_FACTOR = 2
WEIGHT_DECAY = 0.0005
WEIGHT_DECAY_BIAS = 0.
D_lr = 1e-4
iter_sche = [10000, 20000, 30000]

train_batch_size = 8
val_batch_size = 16

log_dir = '../logdir'
show_dir = '../showdir'
model_dir = '../models'
data_folder = '/app/SYSU-MM01'

model_path = os.path.join(model_dir, 'latest')
save_steps = 5000
latest_steps = 100
val_step = 200

num_workers = 4
num_gpu = 1
device_id = '1'
num_classes = 296
test_times = 10 # official setting

# for showing logger
logger = logging.getLogger('train')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


############################# Hyper-parameters ################################
alpha = 1.0
beta = 1.0
at_margin = 1 

pixel_mean = [0.485, 0.456, 0.406]
pixel_std = [0.229, 0.224, 0.225]
inp_size = [384, 128]

# transforms

transforms_list = transforms.Compose([transforms.RectScale(*inp_size),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.Pad(10),
                                      transforms.RandomCrop(inp_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=pixel_mean,
                                                           std=pixel_std),
                                      transforms.RandomErasing(probability=0.5, mean=pixel_mean)])

test_transforms_list = transforms.Compose([
    transforms.RectScale(*inp_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=pixel_mean,
                         std=pixel_std)])

