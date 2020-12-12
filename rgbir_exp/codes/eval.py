"""
Angular Triplet Loss
YE, Hanrong et al, Bi-directional Exponential Angular Triplet Loss for RGB-Infrared Person Re-Identification
"""
from __future__ import print_function, absolute_import
import numpy as np
import copy
from collections import defaultdict
import sys
import torch
import matplotlib.pyplot as plt
import pickle


def test(feature_generators, queryloader, galleryloader, use_gpu = True, ranks=[1, 5, 10, 20]):
    if type(feature_generators) is list:
        feature_generator_rgb = feature_generators[0]
        feature_generator_ir = feature_generators[1]

    else:
        feature_generator_rgb = feature_generators
        feature_generator_ir = feature_generators
    feature_generator_rgb.eval()
    feature_generator_ir.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
            if use_gpu: imgs = imgs.cuda()
            features = feature_generator_ir(imgs) # query features # use fi   
            features = features.data#.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        #end = time.time()
        for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
            if use_gpu: imgs = imgs.cuda()

            features = feature_generator_rgb(imgs)
            features = features.data#.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)

        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))


    qf = qf.view(qf.size(0),-1)
    gf = gf.view(gf.size(0),-1)

    # see norm
    q_norms = qf.norm(dim=1)
    print('q_norms:')
    print(q_norms)

    g_norms = gf.norm(dim=1)
    print('g_norms:')
    print(g_norms)

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.cpu().numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids) # use_metric_cuhk03=args.use_metric_cuhk03)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
    print("------------------")

    return cmc, mAP 

def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with SYSU metric
    Key: for each query identity in camera 3, its gallery images from camera 2 view are discarded.
    """
    
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)

    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (q_camid == 3) & (g_camids[order] == 2)
        keep = np.invert(remove)
        

        if(not q_idx):
            print('Query ID',q_pid)
            for g_idx in range(20):
                print('Gallery ID Rank #', g_idx ,' : ', g_pids[order[g_idx]], 'distance : ', distmat[q_idx][order[g_idx]])

        # compute cmc curve
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]

        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

