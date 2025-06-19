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
import torch.optim as optim
# from model import *
import ipdb
import copy

from loss import *
from utils import *
from retrain_model import Autoencoder

import torch



    
def exp_lr_scheduler(optimizer, epoch, init_lr, lr_decay, decay_rate):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    # 每四次epoch调整一下lr，将lr减半
    lr = init_lr * (decay_rate ** (epoch // lr_decay))  # *是乘法，**是乘方，/是浮点除法，//是整数除法，%是取余数

    if epoch % lr_decay == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # 返回改变了学习率的optimizer
    return optimizer



def retrain_cls_relation(model, prototypes, proto_labels, client_ids, anchors, anchors_labels, weights, vars_list, indicators, n_classes, args, round, device):
    
    if round < 5:
        init_lr = 1e-1
    else:
        init_lr = 1e-2

    model.to(device)
    lr_decay = 45
    decay_rate = 0.1
        
    prototypes = prototypes.to(device)
    proto_labels = proto_labels.to(device)
    client_ids = client_ids.to(device)
    weights = weights.to(device)
    anchors = calculate_class_centers(anchors, anchors_labels)
    anchors = anchors.to(device)
    print(prototypes.shape)    
#     optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=init_lr, weight_decay=args.reg)
#     criterion = nn.CrossEntropyLoss().to(device)
    criterion = WeightedCrossEntropyLoss().to(device)
    Relation_Local = Relation_Local_Loss(args.dist_weight, args.angle_weight, args.mode_local)
    Relation_Complementary = Relation_Complementary_Loss(args.dist_weight, args.angle_weight)
    
    idx_list = np.array(np.arange(len(proto_labels)))
    batch_size = args.re_bs
       
    for epoch in range(100):
        optimizer = exp_lr_scheduler(optimizer, epoch, init_lr, lr_decay, decay_rate)
        random.shuffle(idx_list)
        
        epoch_celoss_collector=[]   
        epoch_relation_local_loss_collector=[] 
        epoch_relation_com_losss_collector=[] 
        epoch_relation_local_mix_loss_collector=[] 
        epoch_relation_com_mix_loss_collector=[] 
        for i in range(len(proto_labels)//batch_size+1):
            if i < len(proto_labels)//batch_size:
                x = prototypes[idx_list[i*batch_size:(i+1)*batch_size]]
                target = proto_labels[idx_list[i*batch_size:(i+1)*batch_size]]             
                batch_client_ids = client_ids[idx_list[i*batch_size:(i+1)*batch_size]] 
                batch_weights = weights[idx_list[i*batch_size:(i+1)*batch_size]] 
                batch_indicators = indicators[idx_list[i*batch_size:(i+1)*batch_size]] 
            else:
                x = prototypes[idx_list[i*batch_size:]]
                target = proto_labels[idx_list[i*batch_size:]]           
                batch_client_ids = client_ids[idx_list[i*batch_size:]]  
                batch_weights = weights[idx_list[i*batch_size:]]  
                batch_indicators = indicators[idx_list[i*batch_size:]]  
                
            ind_ = torch.nonzero(batch_indicators == 0).reshape(-1)
            if len(ind_) > 0:
                vars_i = random.choice(vars_list).cuda()  
                std_vector = torch.sqrt(vars_i)
                mean_vector = torch.zeros(std_vector.size(0)).cuda() 

                expanded_mean = mean_vector.repeat(ind_.shape[0], 1)
                expanded_std = std_vector.repeat(ind_.shape[0], 1)
                random_vectors = torch.normal(mean=expanded_mean, std=expanded_std)
                print(random_vectors.shape)
#                 ipdb.set_trace()
                x[ind_] = x[ind_] + random_vectors.cuda() 
                
            optimizer.zero_grad()
            target = target.long()                
            
            if args.re_phase == 'p1':
                feats, out = model(x)
                celoss = criterion(out, target, batch_weights)
                loss = celoss 
                epoch_celoss_collector.append(celoss.data)   
                epoch_relation_local_loss_collector.append(0) 
                epoch_relation_com_losss_collector.append(0) 
                epoch_relation_local_mix_loss_collector.append(0) 
                epoch_relation_com_mix_loss_collector.append(0) 
                
            elif args.re_phase == 'p2':
                # CE+Local

                feats, out = model(x)
                celoss = criterion(out, target)
                relation_local_loss = Relation_Local(feats, anchors[target], batch_client_ids, args)                
                loss = celoss + args.re_mu * relation_local_loss
                epoch_celoss_collector.append(celoss.data)   
                epoch_relation_local_loss_collector.append(relation_local_loss.data) 
                epoch_relation_com_losss_collector.append(0) 
                epoch_relation_local_mix_loss_collector.append(0) 
                epoch_relation_com_mix_loss_collector.append(0)                 
                
            elif args.re_phase == 'p3':
                # CE+Com
                feats, out = model(x)
                celoss = criterion(out, target)
                relation_com_loss = Relation_Complementary(feats, anchors[target], args)
                loss = celoss + args.re_mu * relation_com_loss  
                epoch_celoss_collector.append(celoss.data)   
                epoch_relation_local_loss_collector.append(0) 
                epoch_relation_com_losss_collector.append(relation_com_loss.data) 
                epoch_relation_local_mix_loss_collector.append(0) 
                epoch_relation_com_mix_loss_collector.append(0) 
                
            elif args.re_phase == 'p4':
                # CE+Local_mix
                
                feats, out = model(x)
                celoss = criterion(out, target) 
                
                if args.mode_mix == 'batch_mix':
                    indices = torch.randperm(x.size(0))
                    x_options = x[indices]
                    options_target = target[indices]
                elif args.mode_mix == 'client_mix':
                    x_options, options_target = in_client_prototype_selection(x, target, prototypes, proto_labels, client_ids, batch_client_ids, mode=args.mode_sample)                     
                mixed_x, target, options_target, lam = mixup_data(x, target, x_options, options_target, alpha=args.re_alpha)               
                feats_mix, _ = model(mixed_x)
                relation_local_mix_loss = lam * Relation_Local(feats_mix, anchors[target], batch_client_ids, args) + (1 - lam) * Relation_Local(feats_mix, anchors[options_target], batch_client_ids, args)
                
                loss = celoss + args.re_mu * relation_local_mix_loss
                epoch_celoss_collector.append(celoss.data)   
                epoch_relation_local_loss_collector.append(0) 
                epoch_relation_com_losss_collector.append(0) 
                epoch_relation_local_mix_loss_collector.append(relation_local_mix_loss.data) 
                epoch_relation_com_mix_loss_collector.append(0) 
                
            elif args.re_phase == 'p5':
                # 只需要CE+Com_mix
                feats, out = model(x)
                celoss = criterion(out, target) 
                
                if args.mode_mix == 'batch_mix':
                    indices = torch.randperm(x.size(0))
                    x_options = x[indices]
                    options_target = target[indices]
                elif args.mode_mix == 'client_mix':
                    x_options, options_target = in_client_prototype_selection(x, target, prototypes, proto_labels, client_ids, batch_client_ids, mode=args.mode_sample)                     
                mixed_x, target, options_target, lam = mixup_data(x, target, x_options, options_target, alpha=args.re_alpha)               
                feats_mix, _ = model(mixed_x)
                relation_com_mix_loss = lam * Relation_Complementary(feats_mix, anchors[target], args) + (1 - lam) * Relation_Complementary(feats_mix, anchors[options_target], args)
                
                loss = celoss + args.re_mu * relation_com_mix_loss
                
                epoch_celoss_collector.append(celoss.data)   
                epoch_relation_local_loss_collector.append(0) 
                epoch_relation_com_losss_collector.append(0) 
                epoch_relation_local_mix_loss_collector.append(0) 
                epoch_relation_com_mix_loss_collector.append(relation_com_mix_loss.data)                          
                
            elif args.re_phase == 'p6':
                # 只需要CE+Local+Com
                feats, out = model(x)
                celoss = criterion(out, target)
                
                relation_local_loss = Relation_Local(feats, anchors[target], batch_client_ids, args)
                relation_com_loss = Relation_Complementary(feats, anchors[target], args)
                loss = celoss + args.re_mu * relation_local_loss + args.re_beta * relation_com_loss
                epoch_celoss_collector.append(celoss.data)   
                epoch_relation_local_loss_collector.append(relation_local_loss.data) 
                epoch_relation_com_losss_collector.append(relation_com_loss.data) 
                epoch_relation_local_mix_loss_collector.append(0) 
                epoch_relation_com_mix_loss_collector.append(0)                 
                               
            elif args.re_phase == 'p7':
                # 只需要CE+Local_mix+Com_mix
                feats, out = model(x)
                celoss = criterion(out, target) 
                
                if args.mode_mix == 'batch_mix':
                    indices = torch.randperm(x.size(0))
                    x_options = x[indices]
                    options_target = target[indices]
                elif args.mode_mix == 'client_mix':
                    x_options, options_target = in_client_prototype_selection(x, target, prototypes, proto_labels, client_ids, batch_client_ids, mode=args.mode_sample)             
            
                mixed_x, target, options_target, lam = mixup_data(x, target, x_options, options_target, alpha=args.re_alpha)               
                feats_mix, _ = model(mixed_x)
                
                relation_local_mix_loss = lam * Relation_Local(feats_mix, anchors[target], batch_client_ids, args) + (1 - lam) * Relation_Local(feats_mix, anchors[options_target], batch_client_ids, args)
                relation_com_mix_loss = lam * Relation_Complementary(feats_mix, anchors[target], args) + (1 - lam) * Relation_Complementary(feats_mix, anchors[options_target], args)
                
                loss = celoss + args.re_mu * relation_local_mix_loss + args.re_beta * relation_com_mix_loss
                
                epoch_celoss_collector.append(celoss.data)   
                epoch_relation_local_loss_collector.append(0) 
                epoch_relation_com_losss_collector.append(0) 
                epoch_relation_local_mix_loss_collector.append(relation_local_mix_loss.data) 
                epoch_relation_com_mix_loss_collector.append(relation_com_mix_loss.data) 
                 
            loss.backward()
            optimizer.step()
#         ipdb.set_trace()
        print(epoch, sum(epoch_celoss_collector)/len(epoch_celoss_collector), sum(epoch_relation_local_loss_collector)/len(epoch_relation_local_loss_collector), sum(epoch_relation_com_losss_collector)/len(epoch_relation_com_losss_collector), sum(epoch_relation_local_mix_loss_collector)/len(epoch_relation_local_mix_loss_collector), sum(epoch_relation_com_mix_loss_collector)/len(epoch_relation_com_mix_loss_collector))

    return model

def generate_random_vectors(var_vectors, k=1):
    random_vectors = []
    for _ in range(k):
        for vars_i in var_vectors:
            std_vector = torch.sqrt(vars_i)  # 计算标准差
            mean_vector = torch.zeros(std_vector.size(0)).cuda()  # 定义均值向量

            # 根据均值和标准差生成随机向量
            random_vector = torch.normal(mean=mean_vector, std=std_vector)
            random_vectors.append(random_vector)

    # 将列表转换为矩阵
#     ipdb.set_trace()
    random_matrix = torch.stack(random_vectors)
    
    return random_matrix



def retrain_cls_relation2(model, prototypes, proto_labels, client_ids, anchors, anchors_labels, weights, vars_list, indicators, n_classes, args, round, device):
    
    if round < 5:
        init_lr = 1e-2
    else:
        init_lr = 5e-3

    model.to(device)
    lr_decay = 40
    decay_rate = 0.1
        
    prototypes = prototypes.to(device)
    proto_labels = proto_labels.to(device)
    client_ids = client_ids.to(device)
    weights = weights.to(device)
    vars_list = vars_list.to(device)
    anchors = calculate_class_centers(anchors, anchors_labels)
    anchors = anchors.to(device)
    print(prototypes.shape)    
#     optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=init_lr, weight_decay=args.reg)
#     criterion = nn.CrossEntropyLoss().to(device)
    criterion = WeightedCrossEntropyLoss().to(device)
    Relation_Local = Relation_Local_Loss(args.dist_weight, args.angle_weight, args.mode_local)
    Relation_Complementary = Relation_Complementary_Loss(args.dist_weight, args.angle_weight)
    
    idx_list = np.array(np.arange(len(proto_labels)))
    batch_size = args.re_bs
       
    for epoch in range(100):
        optimizer = exp_lr_scheduler(optimizer, epoch, init_lr, lr_decay, decay_rate)
        random.shuffle(idx_list)
        
        epoch_celoss_collector=[]   
        epoch_relation_local_loss_collector=[] 
        epoch_relation_com_losss_collector=[] 
        epoch_relation_local_mix_loss_collector=[] 
        epoch_relation_com_mix_loss_collector=[] 
        for i in range(len(proto_labels)//batch_size+1):
            if i < len(proto_labels)//batch_size:
                x = prototypes[idx_list[i*batch_size:(i+1)*batch_size]]
                target = proto_labels[idx_list[i*batch_size:(i+1)*batch_size]]             
                batch_client_ids = client_ids[idx_list[i*batch_size:(i+1)*batch_size]] 
                batch_weights = weights[idx_list[i*batch_size:(i+1)*batch_size]] 
                batch_indicators = indicators[idx_list[i*batch_size:(i+1)*batch_size]] 
            else:
                x = prototypes[idx_list[i*batch_size:]]
                target = proto_labels[idx_list[i*batch_size:]]           
                batch_client_ids = client_ids[idx_list[i*batch_size:]]  
                batch_weights = weights[idx_list[i*batch_size:]]  
                batch_indicators = indicators[idx_list[i*batch_size:]]  
                
            optimizer.zero_grad()
            target = target.long()                
            
            if args.re_phase == 'p1':
                feats, out = model(x)
                celoss = criterion(out, target, batch_weights)
                loss = celoss 
                epoch_celoss_collector.append(celoss.data)   
                epoch_relation_local_loss_collector.append(0) 
                epoch_relation_com_losss_collector.append(0) 
                epoch_relation_local_mix_loss_collector.append(0) 
                epoch_relation_com_mix_loss_collector.append(0) 
                
            elif args.re_phase == 'p2':
                batch_vars = vars_list[batch_client_ids]
                random_vectors = generate_random_vectors(batch_vars).cuda() 
                x_ = x + random_vectors                
                x_all = torch.cat([x, x_], dim=0)
                batch_weights = torch.cat([batch_weights, batch_weights])
                target_all = torch.cat([target, target])
                feats, out_all = model(x_all)
                celoss = criterion(out_all, target_all, batch_weights)
                loss = celoss 
                epoch_celoss_collector.append(celoss.data)   
                epoch_relation_local_loss_collector.append(0) 
                epoch_relation_com_losss_collector.append(0) 
                epoch_relation_local_mix_loss_collector.append(0) 
                epoch_relation_com_mix_loss_collector.append(0) 
                
            elif args.re_phase == 'p3':
                # CE+Local
                x_all = torch.cat([x, x_], dim=0)
                target_all = torch.cat([target, target])
                feats_all, out_all = model(x_all)
                batch_client_ids = torch.cat([batch_client_ids, batch_client_ids])
                celoss = criterion(out_all, target_all, batch_weights)
                relation_local_loss = Relation_Local(feats_all, anchors[target_all], batch_client_ids, args)                
                loss = celoss + args.re_mu * relation_local_loss
                epoch_celoss_collector.append(celoss.data)   
                epoch_relation_local_loss_collector.append(relation_local_loss.data) 
                epoch_relation_com_losss_collector.append(0) 
                epoch_relation_local_mix_loss_collector.append(0) 
                epoch_relation_com_mix_loss_collector.append(0)                 
                
            elif args.re_phase == 'p4':
                # CE+Com
                feats, out = model(x)
                celoss = criterion(out, target)
                relation_com_loss = Relation_Complementary(feats, anchors[target], args)
                loss = celoss + args.re_mu * relation_com_loss  
                epoch_celoss_collector.append(celoss.data)   
                epoch_relation_local_loss_collector.append(0) 
                epoch_relation_com_losss_collector.append(relation_com_loss.data) 
                epoch_relation_local_mix_loss_collector.append(0) 
                epoch_relation_com_mix_loss_collector.append(0) 
                                    
                
            elif args.re_phase == 'p5':
                # 只需要CE+Local+Com
                feats, out = model(x)
                celoss = criterion(out, target)
                
                relation_local_loss = Relation_Local(feats, anchors[target], batch_client_ids, args)
                relation_com_loss = Relation_Complementary(feats, anchors[target], args)
                loss = celoss + args.re_mu * relation_local_loss + args.re_beta * relation_com_loss
                epoch_celoss_collector.append(celoss.data)   
                epoch_relation_local_loss_collector.append(relation_local_loss.data) 
                epoch_relation_com_losss_collector.append(relation_com_loss.data) 
                epoch_relation_local_mix_loss_collector.append(0) 
                epoch_relation_com_mix_loss_collector.append(0)                 
                 
            loss.backward()
            optimizer.step()
#         ipdb.set_trace()
        print(epoch, sum(epoch_celoss_collector)/len(epoch_celoss_collector), sum(epoch_relation_local_loss_collector)/len(epoch_relation_local_loss_collector), sum(epoch_relation_com_losss_collector)/len(epoch_relation_com_losss_collector), sum(epoch_relation_local_mix_loss_collector)/len(epoch_relation_local_mix_loss_collector), sum(epoch_relation_com_mix_loss_collector)/len(epoch_relation_com_mix_loss_collector))

    return model


def retrain_cls_ablation(model, prototypes, proto_labels, client_ids, anchors, anchors_labels, n_classes, args, round, device):
    
    if round < 10:
        init_lr = 1e-1
    else:
        init_lr = 1e-2

    model.to(device)
    lr_decay = 40
    decay_rate = 0.1
        
    prototypes = prototypes.to(device)
    proto_labels = proto_labels.to(device)
    client_ids = client_ids.to(device)
    anchors = calculate_class_centers(anchors, anchors_labels)
    anchors = anchors.to(device)
    
#     automodel = Autoencoder(anchors.shape[1], 256, 256).to(device)
#     automodel = train_anchors(automodel, anchors, 91, 1e-4)
#     anchors = automodel(anchors)    
    
#     optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=init_lr, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)
    
    local_edge_criterion = Local_Edge_Loss(args.mode_local)
    local_angle_criterion = Local_Angle_Loss(args.mode_local)
    com_edge_criterion = Com_Edge_Loss()
    com_angle_criterion = Com_Angle_Loss()
    
    idx_list = np.array(np.arange(len(proto_labels)))
    batch_size = args.re_bs
       
    for epoch in range(100):
        optimizer = exp_lr_scheduler(optimizer, epoch, init_lr, lr_decay, decay_rate)
        random.shuffle(idx_list)
        
        epoch_celoss_collector=[]   
        epoch_local_edge_loss_collector=[] 
        epoch_local_angle_loss_collector=[] 
        epoch_com_edge_loss_collector=[] 
        epoch_com_angle_loss_collector=[] 
        for i in range(len(proto_labels)//batch_size + 1):
            if i < len(proto_labels)//batch_size :
                x = prototypes[idx_list[i*batch_size:(i+1)*batch_size]]
                target = proto_labels[idx_list[i*batch_size:(i+1)*batch_size]]             
                batch_client_ids = client_ids[idx_list[i*batch_size:(i+1)*batch_size]] 
            else:
                x = prototypes[idx_list[i*batch_size:]]
                target = proto_labels[idx_list[i*batch_size:]]             
                batch_client_ids = client_ids[idx_list[i*batch_size:]]                 

            optimizer.zero_grad()
            target = target.long()         
                        
            if args.re_phase == 'p1':
                # 只需要CE
                feats, out = model(x)
                celoss = criterion(out, target)
                loss = celoss 
                epoch_celoss_collector.append(celoss.data)   
                epoch_local_edge_loss_collector.append(0) 
                epoch_local_angle_loss_collector.append(0) 
                epoch_com_edge_loss_collector.append(0) 
                epoch_com_angle_loss_collector.append(0) 
                
            elif args.re_phase == 'p2':
                # CE+Local_edge
                feats, out = model(x)
                celoss = criterion(out, target)
                local_edge_loss = local_edge_criterion(feats, anchors[target], batch_client_ids, args)                
                loss = celoss + args.re_mu * local_edge_loss
                epoch_celoss_collector.append(celoss.data)   
                epoch_local_edge_loss_collector.append(local_edge_loss.data) 
                epoch_local_angle_loss_collector.append(0) 
                epoch_com_edge_loss_collector.append(0) 
                epoch_com_angle_loss_collector.append(0)                
                
            elif args.re_phase == 'p3':
                # CE+Local_angle
                feats, out = model(x)
                celoss = criterion(out, target)
                local_angle_loss = local_angle_criterion(feats, anchors[target], batch_client_ids, args)
                loss = celoss + args.re_mu * local_angle_loss  
                epoch_celoss_collector.append(celoss.data)   
                epoch_local_edge_loss_collector.append(0) 
                epoch_local_angle_loss_collector.append(local_angle_loss.data) 
                epoch_com_edge_loss_collector.append(0) 
                epoch_com_angle_loss_collector.append(0) 
                
            elif args.re_phase == 'p4':
                # CE+Com_edge
                feats, out = model(x)
                celoss = criterion(out, target) 
                com_edge_loss = com_edge_criterion(feats, anchors[target], args)                
                loss = celoss + args.re_mu * com_edge_loss
                epoch_celoss_collector.append(celoss.data)   
                epoch_local_edge_loss_collector.append(0) 
                epoch_local_angle_loss_collector.append(0) 
                epoch_com_edge_loss_collector.append(com_edge_loss.data) 
                epoch_com_angle_loss_collector.append(0) 
                
            elif args.re_phase == 'p5':
                # 只需要 CE+Com_angle
                feats, out = model(x)
                celoss = criterion(out, target) 
                com_angle_loss = com_angle_criterion(feats, anchors[target], args)                
                loss = celoss + args.re_mu * com_angle_loss
                
                epoch_celoss_collector.append(celoss.data)   
                epoch_local_edge_loss_collector.append(0) 
                epoch_local_angle_loss_collector.append(0) 
                epoch_com_edge_loss_collector.append(0) 
                epoch_com_angle_loss_collector.append(com_angle_loss.data)                        
                
            elif args.re_phase == 'p6':
                # 只需要CE+Local_edge+Local_angle
                feats, out = model(x)
                celoss = criterion(out, target)
                local_edge_loss = local_edge_criterion(feats, anchors[target], batch_client_ids, args)  
                local_angle_loss = local_angle_criterion(feats, anchors[target], batch_client_ids, args)
                loss = celoss + args.re_mu * local_edge_loss + args.re_beta * local_angle_loss
                epoch_celoss_collector.append(celoss.data)   
                epoch_local_edge_loss_collector.append(local_edge_loss.data) 
                epoch_local_angle_loss_collector.append(local_angle_loss.data) 
                epoch_com_edge_loss_collector.append(0) 
                epoch_com_angle_loss_collector.append(0)               
                               
            elif args.re_phase == 'p7':
                # 只需要CE+Local_edge+Com_edge
                feats, out = model(x)
                celoss = criterion(out, target) 
                local_edge_loss = local_edge_criterion(feats, anchors[target], batch_client_ids, args)  
                com_edge_loss = com_edge_criterion(feats, anchors[target], args)                
                loss = celoss + args.re_mu * local_edge_loss + args.re_beta * com_edge_loss
                
                epoch_celoss_collector.append(celoss.data)   
                epoch_local_edge_loss_collector.append(local_edge_loss.data) 
                epoch_local_angle_loss_collector.append(0) 
                epoch_com_edge_loss_collector.append(com_edge_loss.data) 
                epoch_com_angle_loss_collector.append(0) 

            elif args.re_phase == 'p8':
                # 只需要CE+Com_edge+Com_angle     
                feats, out = model(x)
                celoss = criterion(out, target) 
                com_edge_loss = com_edge_criterion(feats, anchors[target], args) 
                com_angle_loss = com_angle_criterion(feats, anchors[target], args)
                
                loss = celoss + args.re_mu * com_edge_loss + args.re_beta * com_angle_loss
                
                epoch_celoss_collector.append(celoss.data)   
                epoch_local_edge_loss_collector.append(0) 
                epoch_local_angle_loss_collector.append(0) 
                epoch_com_edge_loss_collector.append(com_edge_loss.data) 
                epoch_com_angle_loss_collector.append(com_angle_loss.data)                
    
            elif args.re_phase == 'p9':
                # 只需要CE+Local_angle+Com_angle 
                feats, out = model(x)
                celoss = criterion(out, target) 
                local_angle_loss = local_angle_criterion(feats, anchors[target], batch_client_ids, args)
                com_angle_loss = com_angle_criterion(feats, anchors[target], args)
                
                loss = celoss + args.re_mu * local_angle_loss + args.re_beta * com_angle_loss
                
                epoch_celoss_collector.append(celoss.data)   
                epoch_local_edge_loss_collector.append(0) 
                epoch_local_angle_loss_collector.append(local_angle_loss.data) 
                epoch_com_edge_loss_collector.append(0) 
                epoch_com_angle_loss_collector.append(com_angle_loss.data)               
                                
                
            elif args.re_phase == 'p10':
                # 只需要CE+Local_edge+Local_angle+Com_edge+Com_angle   
                feats, out = model(x)
                celoss = criterion(out, target) 
                local_edge_loss = local_edge_criterion(feats, anchors[target], batch_client_ids, args)  
                local_angle_loss = local_angle_criterion(feats, anchors[target], batch_client_ids, args)
                com_edge_loss = com_edge_criterion(feats, anchors[target], args) 
                com_angle_loss = com_angle_criterion(feats, anchors[target], args)                
                loss = celoss + args.re_mu * local_edge_loss + args.re_beta * local_angle_loss + args.re_gamma * com_edge_loss + args.re_lambda * com_angle_loss                
                epoch_celoss_collector.append(celoss.data)   
                epoch_local_edge_loss_collector.append(local_edge_loss.data) 
                epoch_local_angle_loss_collector.append(local_angle_loss.data) 
                epoch_com_edge_loss_collector.append(com_edge_loss.data) 
                epoch_com_angle_loss_collector.append(com_angle_loss.data)                    
                
                
            loss.backward()
            optimizer.step()
#         ipdb.set_trace()
        print(epoch, sum(epoch_celoss_collector)/len(epoch_celoss_collector), sum(epoch_local_edge_loss_collector)/len(epoch_local_edge_loss_collector), sum(epoch_local_angle_loss_collector)/len(epoch_local_angle_loss_collector), sum(epoch_com_edge_loss_collector)/len(epoch_com_edge_loss_collector), sum(epoch_com_angle_loss_collector)/len(epoch_com_angle_loss_collector))

    return model





def retrain_cls_relation__(model, prototypes, proto_labels, client_ids, anchors, anchors_labels, n_classes, args, round, device):
    
    if round < 5:
        init_lr = 1e-1
    else:
        init_lr = 1e-2

    model.to(device)
    lr_decay = 40
    decay_rate = 0.1
        
    prototypes = prototypes.to(device)
    proto_labels = proto_labels.to(device)
    client_ids = client_ids.to(device)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=init_lr, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)
    
    relation_criterion = Relation_Loss(args.dist_weight, args.angle_weight)
    
    idx_list = np.array(np.arange(len(proto_labels)))
    batch_size = args.re_bs
       
    for epoch in range(100):
        optimizer = exp_lr_scheduler(optimizer, epoch, init_lr, lr_decay, decay_rate)
        random.shuffle(idx_list)
        
        epoch_loss_collector=[]   
        epoch_mixloss_collector=[] 
        epoch_relationloss_collector=[] 
        epoch_relationlossmix_collector=[] 
        for i in range(len(proto_labels)//batch_size + 1):
            if i < len(proto_labels)//batch_size :
                x = prototypes[idx_list[i*batch_size:(i+1)*batch_size]]
                target = proto_labels[idx_list[i*batch_size:(i+1)*batch_size]]             
                batch_client_ids = client_ids[idx_list[i*batch_size:(i+1)*batch_size]] 
            else:
                x = prototypes[idx_list[i*batch_size:]]
                target = proto_labels[idx_list[i*batch_size:]]             
                batch_client_ids = client_ids[idx_list[i*batch_size:]]                 

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()         
                        
            if args.re_phase == 'p1':
                # 只需要CE
                fexats, out = model(x)
                celoss = criterion(out, target)
                loss = celoss # + args.re_mu * relationloss + args.re_beta * mixloss
                epoch_relationloss_collector.append(0)
                epoch_mixloss_collector.append(0)
                epoch_loss_collector.append(celoss.data)
                epoch_relationlossmix_collector.append(0)
            elif args.re_phase == 'p2':
                # 只需要mixup
#                 x_options, options_target = in_client_prototype_selection(x, target, prototypes, proto_labels, client_ids, batch_client_ids, mode=args.mode_sample) 
                indices = torch.randperm(x.size(0))
                x_options = x[indices]
                options_target = target[indices]
            
                mixed_x, target, options_target, lam = mixup_data(x, target, x_options, options_target, alpha=args.re_alpha)
                _, out_mix = model(mixed_x)            
                mixloss = mixup_criterion(criterion, out_mix, target, options_target, lam)
                loss = mixloss
                epoch_relationloss_collector.append(0)
                epoch_mixloss_collector.append(mixloss.data)
                epoch_loss_collector.append(0)
                epoch_relationlossmix_collector.append(0)  
                
            elif args.re_phase == 'p3':
                # 只需要CE+mix
                x_options, options_target = in_client_prototype_selection(x, target, prototypes, proto_labels, client_ids, batch_client_ids, mode=args.mode_sample)  
                x_cat = torch.cat([x, x_options])
                target_cat = torch.cat([target, options_target])
                feats_cat, out_cat = model(x_cat)
                celoss = criterion(out_cat, target_cat)
                mixed_x, target, options_target, lam = mixup_data(x, target, x_options, options_target, alpha=args.re_alpha)
                _, out_mix = model(mixed_x)            
                mixloss = mixup_criterion(criterion, out_mix, target, options_target, lam)                
                loss = celoss + args.re_beta * mixloss
                epoch_relationloss_collector.append(0)
                epoch_mixloss_collector.append(mixloss.data)
                epoch_loss_collector.append(celoss.data)
                epoch_relationlossmix_collector.append(0)               
                
            elif args.re_phase == 'p4':
                # CE+RGM
                feats, out = model(x)
                celoss = criterion(out, target) 
                relationloss = relation_criterion(feats, anchors[target], batch_client_ids, args) 
                loss = celoss + args.re_mu * relationloss
                epoch_relationloss_collector.append(relationloss.data)
                epoch_mixloss_collector.append(0)
                epoch_loss_collector.append(celoss.data)
                epoch_relationlossmix_collector.append(0)
                
            elif args.re_phase == 'p5':
                # 只需要CE+RGM_mix
                x_options, options_target = in_client_prototype_selection(x, target, prototypes, proto_labels, client_ids, batch_client_ids, mode=args.mode_sample)  
                x_cat = torch.cat([x, x_options])
                target_cat = torch.cat([target, options_target])
                feats_cat, out_cat = model(x)
                celoss = criterion(out_cat, target)    
                mixed_x, target, options_target, lam = mixup_data(x, target, x_options, options_target, alpha=args.re_alpha)
                feats_mix, _ = model(mixed_x) 
                
                relationloss_mix = lam * relation_criterion(feats_mix, anchors[target], batch_client_ids, args) + (1 - lam) * relation_criterion(feats_mix, anchors[options_target], batch_client_ids, args)                
                loss = celoss + args.re_mu * relationloss_mix
                epoch_relationloss_collector.append(0)
                epoch_mixloss_collector.append(0)
                epoch_loss_collector.append(celoss.data)
                epoch_relationlossmix_collector.append(relationloss_mix.data)                           
                
            elif args.re_phase == 'p6':
                # 只需要CE_mix+RGM
                x_options, options_target = in_client_prototype_selection(x, target, prototypes, proto_labels, client_ids, batch_client_ids, mode=args.mode_sample)  
                mixed_x, target, options_target, lam = mixup_data(x, target, x_options, options_target, alpha=args.re_alpha)
                feats_mix, out_mix = model(mixed_x)         
                mixloss = mixup_criterion(criterion, out_mix, target, options_target, lam)
                x_cat = torch.cat([x, x_options])
                target_cat = torch.cat([target, options_target])  
                batch_client_ids = torch.cat([batch_client_ids, batch_client_ids]) 
                feats_cat, _ = model(x_cat)
                relationloss_mix = relation_criterion(feats_cat, anchors[target_cat], batch_client_ids, args) 
                loss = args.re_beta * mixloss + args.re_mu * relationloss_mix
                epoch_relationloss_collector.append(0)
                epoch_mixloss_collector.append(mixloss.data)
                epoch_loss_collector.append(0)
                epoch_relationlossmix_collector.append(relationloss_mix.data)                
                                
            elif args.re_phase == 'p7':
                # 只需要CE_mix+RGM_mix
                x_options, options_target = in_client_prototype_selection(x, target, prototypes, proto_labels, client_ids, batch_client_ids, mode=args.mode_sample)  
                mixed_x, target, options_target, lam = mixup_data(x, target, x_options, options_target, alpha=args.re_alpha)
                feats_mix, out_mix = model(mixed_x)         
                mixloss = mixup_criterion(criterion, out_mix, target, options_target, lam)                
                
                relationloss_mix = lam * relation_criterion(feats_mix, anchors[target], batch_client_ids, args) + (1 - lam) * relation_criterion(feats_mix, anchors[options_target], batch_client_ids, args)
                loss = args.re_beta * mixloss + args.re_mu * relationloss_mix
                epoch_relationloss_collector.append(0)
                epoch_mixloss_collector.append(mixloss.data)
                epoch_loss_collector.append(0)
                epoch_relationlossmix_collector.append(relationloss_mix.data)
                
            elif args.re_phase == 'p8':
                # CE+CE_mix+RGM_mix            
                x_options, options_target = in_client_prototype_selection(x, target, prototypes, proto_labels, client_ids, batch_client_ids, mode=args.mode_sample)  
                mixed_x, target, options_target, lam = mixup_data(x, target, x_options, options_target, alpha=args.re_alpha)
                feats_mix, out_mix = model(mixed_x)         
                mixloss = mixup_criterion(criterion, out_mix, target, options_target, lam)                
                
                relationloss = lam * relation_criterion(feats_mix, anchors[target], batch_client_ids, args) + (1 - lam) * relation_criterion(feats_mix, anchors[options_target], batch_client_ids, args)                
                x_cat = torch.cat([x, x_options])
                target_cat = torch.cat([target, options_target])
                feats_cat, out_cat = model(x_cat)
                celoss = criterion(out_cat, target_cat)   
                loss = celoss + args.re_beta * mixloss + args.re_mu * relationloss
                epoch_relationloss_collector.append(0)
                epoch_mixloss_collector.append(mixloss.data)
                epoch_loss_collector.append(celoss.data)
                epoch_relationlossmix_collector.append(relationloss.data)
                
            elif args.re_phase == 'p9':
                # CE+CE_mix+RGM            
                x_options, options_target = in_client_prototype_selection(x, target, prototypes, proto_labels, client_ids, batch_client_ids, mode=args.mode_sample)  
                mixed_x, target, options_target, lam = mixup_data(x, target, x_options, options_target, alpha=args.re_alpha)
                feats_mix, out_mix = model(mixed_x)         
                mixloss = mixup_criterion(criterion, out_mix, target, options_target, lam)                
                x_cat = torch.cat([x, x_options])
                target_cat = torch.cat([target, options_target])
                feats_cat, out_cat = model(x_cat)
                celoss = criterion(out_cat, target_cat) 

                batch_client_ids = torch.cat([batch_client_ids, batch_client_ids]) 
                relationloss = relation_criterion(feats_cat, anchors[target_cat], batch_client_ids, args)                   
                loss = celoss + args.re_beta * mixloss + args.re_mu * relationloss                
                epoch_relationloss_collector.append(relationloss.data)
                epoch_mixloss_collector.append(mixloss.data)
                epoch_loss_collector.append(celoss.data)
                epoch_relationlossmix_collector.append(0)                
                
            elif args.re_phase == 'p10':
                # CE+CE_mix+RGM+RGM_mix              
                x_options, options_target = in_client_prototype_selection(x, target, prototypes, proto_labels, client_ids, batch_client_ids, mode=args.mode_sample)  
                mixed_x, target, options_target, lam = mixup_data(x, target, x_options, options_target, alpha=args.re_alpha)
                feats_mix, out_mix = model(mixed_x)         
                mixloss = mixup_criterion(criterion, out_mix, target, options_target, lam)                
                x_cat = torch.cat([x, x_options])
                target_cat = torch.cat([target, options_target])
                feats_cat, out_cat = model(x_cat)
                celoss = criterion(out_cat, target_cat) 

                batch_client_ids_ = torch.cat([batch_client_ids, batch_client_ids]) 
                relationloss = relation_criterion(feats_cat, anchors[target_cat], batch_client_ids_, args)   
                
                relationloss_mix = lam * relation_criterion(feats_mix, anchors[target], batch_client_ids, args) + (1 - lam) * relation_criterion(feats_mix, anchors[options_target], batch_client_ids, args)
                loss = celoss + args.re_beta * mixloss + args.re_mu * relationloss + args.re_gamma * relationloss_mix 
                epoch_relationloss_collector.append(relationloss.data)
                epoch_mixloss_collector.append(mixloss.data)
                epoch_loss_collector.append(celoss.data)
                epoch_relationlossmix_collector.append(relationloss_mix.data)

            elif args.re_phase == 'p11':
                # CE+RGM+RGM_mix            
                x_options, options_target = in_client_prototype_selection(x, target, prototypes, proto_labels, client_ids, batch_client_ids, mode=args.mode_sample)  
                mixed_x, target, options_target, lam = mixup_data(x, target, x_options, options_target, alpha=args.re_alpha)
                feats_mix, out_mix = model(mixed_x)                    
                x_cat = torch.cat([x, x_options])
                target_cat = torch.cat([target, options_target])
                feats_cat, out_cat = model(x_cat)
                celoss = criterion(out_cat, target_cat) 
                batch_client_ids_ = torch.cat([batch_client_ids, batch_client_ids]) 
                relationloss = relation_criterion(feats_cat, anchors[target_cat], batch_client_ids_, args)  
                relationloss_mix = lam * relation_criterion(feats_mix, anchors[target], batch_client_ids, args) + (1 - lam) * relation_criterion(feats_mix, anchors[options_target], batch_client_ids, args)
                loss = celoss + args.re_gamma * relationloss_mix + args.re_mu * relationloss                
                epoch_relationloss_collector.append(relationloss.data)
                epoch_mixloss_collector.append(0)
                epoch_loss_collector.append(celoss.data)
                epoch_relationlossmix_collector.append(relationloss_mix.data)    

            elif args.re_phase == 'p12':
                # CE+RGM
                
                Relation_Local = Relation_Local_Loss(args.dist_weight, args.angle_weight)
                Relation_Complementary = Relation_Complementary_Loss(args.dist_weight, args.angle_weight)
                
                feats, out = model(x)
                celoss = criterion(out, target) 
                relation_local_loss = Relation_Local(prototypes, anchors[proto_labels], client_ids, args) 
                relation_complementary_loss = Relation_Complementary(feats, anchors[target], batch_client_ids, args)
                loss = celoss + args.re_mu * (relation_local_loss + relation_complementary_loss)
                epoch_relationloss_collector.append(relation_local_loss.data)
                epoch_mixloss_collector.append(0)
                epoch_loss_collector.append(celoss.data)
                epoch_relationlossmix_collector.append(relation_complementary_loss.data)                
                
                
                
            loss.backward()
            optimizer.step()
#         ipdb.set_trace()
        print(epoch, sum(epoch_loss_collector)/len(epoch_loss_collector), sum(epoch_mixloss_collector)/len(epoch_mixloss_collector), sum(epoch_relationloss_collector)/len(epoch_relationloss_collector),sum(epoch_relationlossmix_collector)/len(epoch_relationlossmix_collector))

    return model



def retrain_cls_relation_(model, prototypes, proto_labels, global_prototypes, n_classes, args, round, device, local_vars=None):
    
    if round < 5:
        init_lr = 1e-1
    else:
        init_lr = 1e-2

    model.to(device)
    lr_decay = 40
    decay_rate = 0.1
        
    prototypes = prototypes.to(device)
    proto_labels = proto_labels.to(device)
    
    random_vectors = generate_random_vectors(local_vars, k=args.re_k).to(device)
    local_protos = torch.cat([prototypes]*args.re_k).to(device) + random_vectors
    local_labels = torch.cat([proto_labels]*args.re_k).to(device)
    
    local_protos = torch.cat([prototypes, local_protos])
    local_labels = torch.cat([proto_labels, local_labels])
    print(proto_labels.shape)
#     ipdb.set_trace() 
#     optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=init_lr, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)
    
    relation_criterion = Relation_Loss(args.dist_weight, args.angle_weight)
    if args.re_phase != 'p5':
        idx_list = np.array(np.arange(len(proto_labels)))
    else:
        idx_list = np.array(np.arange(len(local_labels)))
    batch_size = args.re_bs
       
    for epoch in range(100):
        optimizer = exp_lr_scheduler(optimizer, epoch, init_lr, lr_decay, decay_rate)
        random.shuffle(idx_list)
        
        epoch_loss_collector=[]      
        epoch_relationloss_collector=[] 
        if args.re_phase != 'p5':
            for i in range((len(proto_labels) + batch_size - 1) // batch_size):  # 向上取整计算需要多少批次
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, len(proto_labels))  # 防止索引超出范围
                if start_index < end_index:  # 确保批次不为空
                    x = prototypes[idx_list[start_index:end_index]]
                    target = proto_labels[idx_list[start_index:end_index]]
                    batch_vars = local_vars[idx_list[start_index:end_index]]

                    if args.re_phase == 'p1':
                        optimizer.zero_grad()
                        x.requires_grad = True
                        target.requires_grad = False
                        target = target.long()  
                        # 只需要分类
                        feats, out = model(x)
                        celoss = criterion(out, target)
                        relationloss = torch.tensor(0).cuda()

                    elif args.re_phase == 'p2':
                        optimizer.zero_grad()
                        target = target.long()   
#                         if round > 1:
#                             ipdb.set_trace()
                        # 增广+分类
                        random_vectors = generate_random_vectors(batch_vars, k=args.re_k).cuda()
                        x = torch.cat([x]*args.re_k)
                        target = torch.cat([target]*args.re_k)
                        x.requires_grad = True
                        target.requires_grad = False                        
                        feats, out = model(x+random_vectors)

                        celoss = criterion(out, target)
                        relationloss = torch.tensor(0).cuda()

                    elif args.re_phase == 'p3':
                        optimizer.zero_grad()
                        target = target.long()                        
                        # 增广+分类+原始
                        random_vectors = generate_random_vectors(batch_vars, k=args.re_k).cuda()
                        x_ = torch.cat([x]*args.re_k)
                        x_ = x_ + random_vectors
                        x_all = torch.cat([x, x_], dim=0)
                        target_all = torch.cat([target]*(args.re_k+1))
                        x_all.requires_grad = True
                        target_all.requires_grad = False     
                        feats, out = model(x_all)
                        celoss = criterion(out, target_all)
                        relationloss = torch.tensor(0).cuda()
                    elif args.re_phase == 'p4':
                        optimizer.zero_grad()
                        target = target.long()                           
                        # 增广+分类+原始+关系
                        random_vectors = generate_random_vectors(batch_vars).cuda()
                        x_ = x + random_vectors                
                        x_all = torch.cat([x, x_], dim=0)
                        target_all = torch.cat([target, target])  
                        x_all.requires_grad = True
                        target_all.requires_grad = False                           
                        feats, out = model(x_all)
                        celoss = criterion(out, target_all)
                        relationloss = relation_criterion(feats, global_prototypes[target_all], args)
                    loss = celoss + args.re_mu * relationloss    
                    epoch_loss_collector.append(celoss.data)
                    epoch_relationloss_collector.append(relationloss.data)

                    loss.backward()
                    optimizer.step()
        elif args.re_phase == 'p5':
            for i in range((len(local_labels) + batch_size - 1) // batch_size):  # 向上取整计算需要多少批次
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, len(local_protos))  # 防止索引超出范围
                if end_index - start_index > 1:  # 确保批次不为空
                    x = local_protos[idx_list[start_index:end_index]]
                    target = local_labels[idx_list[start_index:end_index]]

                    optimizer.zero_grad()
                    x.requires_grad = True
                    target.requires_grad = False
                    target = target.long()        
                    feats, out = model(x)
                    celoss = criterion(out, target)
                    relationloss = relation_criterion(feats, global_prototypes[target], args)               
#                     if (torch.isnan(celoss).any()) or (torch.isnan(relationloss).any()):
#                         ipdb.set_trace()
# #                     import ipdb; ipdb.set_trace()
                loss = celoss + args.re_mu * relationloss

                epoch_loss_collector.append(celoss.data)
                epoch_relationloss_collector.append(relationloss.data)

                loss.backward()
                optimizer.step()
        print(epoch, sum(epoch_loss_collector)/len(epoch_loss_collector), sum(epoch_relationloss_collector)/len(epoch_relationloss_collector))

    return model






def retrain_incremental(model, prototypes, proto_labels, client_ids, n_classes, args, round, device):
    
    if round <= 20:
        init_lr = 1e-1
    elif round <= 100:
        init_lr = 1e-2
    else:
        init_lr = 1e-3

    lr_decay = 50
    decay_rate = 0.1

    model.to(device)
    prototypes = prototypes.to(device)
    proto_labels = proto_labels.to(device)
    client_ids = client_ids.to(device)
    anchors = calculate_class_centers(prototypes, proto_labels)
    anchors = anchors.to(device)   
    
    automodel = Autoencoder(anchors.shape[1], 256).to(device)
    automodel = train_anchors(automodel, anchors, 91, 1e-4)
    anchors = automodel(anchors)
       
    cuda = 1
    
    dist_criterion = RkdDistance()
    angle_criterion = RKdAngle()
    
#     optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=init_lr, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)
    
    idx_list = np.array(range(args.n_parties))
#     ipdb.set_trace()
    with torch.no_grad():
        h2, out = model(prototypes)
        pred_label = torch.argmax(out.data, 1)
        total = prototypes.data.size()[0]
        
        correct = (pred_label == proto_labels.data).sum().item()
        print('before', correct)
 
    print('proto_labels', proto_labels.shape)
    
    for idx, client_index in enumerate(idx_list):
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=init_lr, weight_decay=args.reg)
        if client_index == 0:
            # 计算当前客户端的特征表示
            client_prototypes = prototypes[client_ids == client_index]
            client_labels = proto_labels[client_ids == client_index]
            clients_id = client_ids[client_ids == client_index]
     
        else:
            client_prototypes = torch.cat([client_prototypes, prototypes[client_ids == client_index]])
            client_labels = torch.cat([client_labels, proto_labels[client_ids == client_index]])
            clients_id = torch.cat([clients_id, client_ids[client_ids == client_index]])    
        print(client_labels.shape)
        
        batch_size = 25
        for epoch in range(100):
            
            
            optimizer = exp_lr_scheduler(optimizer, epoch, init_lr, lr_decay, decay_rate)
            optimizer.zero_grad()
            # 获取当前要对齐的客户端的原型和标签
            
            client_features, out = model(client_prototypes)
#                 ipdb.set_trace()
            celoss = criterion(out, client_labels)
            distloss = dist_criterion(client_features, client_labels, anchors, clients_id)
            angleloss = angle_criterion(client_features, client_labels, anchors, clients_id) 
            loss = celoss + args.re_mu * distloss + args.re_beta * angleloss

 
            print(epoch, loss.data)
            # 反向传播和优化
            loss.backward()
            optimizer.step()

     
    with torch.no_grad():
        feats, out = model(prototypes)
      
        pred_label = torch.argmax(out.data, 1)
        total = prototypes.data.size()[0]
        correct = (pred_label == proto_labels.data).sum().item()
#         correct_id = torch.nonzero(pred_label == proto_labels.data).reshape(-1)
        
#         protos, labels = gen_proto_global(feats[correct_id], proto_labels[correct_id], n_classes) 
        print('after', correct)

    return model, anchors





