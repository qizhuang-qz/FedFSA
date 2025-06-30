import os
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import random
from sklearn.metrics import confusion_matrix
from torchvision import datasets
from sampling import *
from datasets import *
from utils import *
import ipdb
from sklearn.neighbors import LocalOutlierFactor

def prototype_generation(clip, net_dataidx_map, n_classes, args, outlier=False):
    
    prototypes = {}
    labels = {}
    for i in range(args.n_parties):
        dataidxs = net_dataidx_map[i]
        train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, drop_last=False)
        prototype, label = gen_proto_local(clip, train_dl_local, n_class=n_classes, device='cuda:0')
        prototypes[i] = prototype
        labels[i] = label
        
    if outlier == True:
        prototypes, labels = prototype_outlier(prototypes, labels, n_classes)
#         ipdb.set_trace()
    return prototypes, labels

def prototype_outlier(prototypes, labels, n_classes):
    local_protos = []
    proto_labels = []
    clients_index = []
    for i in range(10):
        local_protos.append(prototypes[i])
        proto_labels.append(labels[i])
        clients_index.append([i]*len(labels[i]))
        
    local_protos = np.concatenate(local_protos)
    proto_labels = np.concatenate(proto_labels)
    clients_index = np.concatenate(clients_index)
#     ipdb.set_trace()
    score = np.zeros(proto_labels.shape[0])
    clf = LocalOutlierFactor(n_neighbors=5, contamination='auto')

    for j in range(n_classes): 
        p_j = np.where(proto_labels == j)[0]
        c_j = clients_index[p_j]
        y_pred = clf.fit_predict(local_protos[p_j])
        inlier_id = np.where(y_pred == 1)[0]

        for k in range(len(p_j)):
            if y_pred[k] == -1:
                prototypes[c_j[k]][j] = np.mean(local_protos[p_j[inlier_id]], axis=0).reshape((1, -1))
    
    return prototypes, labels


def gen_proto_local(net, dataloader, n_class=10, device='cuda:0'):
    feats = []
    labels = []
    net.eval()
    net.apply(fix_bn)
    net.to(device)
    with torch.no_grad():
        for batch_idx, (_, x, target) in enumerate(dataloader):
            x, target = x.to(device), target.to(device)
            feat = net.encode_image(x)

            feats.append(feat)
            labels.extend(target)

    feats = torch.cat(feats).cpu().numpy()
    labels = torch.tensor(labels).cpu().numpy()
#     ipdb.set_trace()
    prototype = np.zeros((n_class, feats.shape[1]))
    label = []
    for i in range(n_class):
        index = np.where(labels == i)[0]
#         ipdb.set_trace()
        if len(index) > 0:
            prototype[i] = np.mean(feats[index], axis=0).reshape((1, -1))
            label.append(i)
    return prototype, label


def generate_text_prototype(clip, clip_model, classes_names, device):
    with torch.no_grad():
        text_inputs = torch.cat([clip.tokenize(f"a photo of {c}.") for c in classes_names]).to(device)
        text_prototypes = clip_model.encode_text(text_inputs)
    return text_prototypes
    
    
    
    
    
    
    
