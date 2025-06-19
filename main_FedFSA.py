import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import logging
import os
import copy
import datetime
import random
from resnet import *
import clip
from model import *
from utils import *
from prototype_generation import *
from retrain_model import *
from re_training import retrain_cls_relation_
import ipdb
import torch.nn.functional as F
from loss import *
from classes_names import *
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='simple-cnn', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used for training')
    parser.add_argument('--partition', type=str, default='noniid', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=10, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=10, help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='communication strategy: fedavg/fedprox')
    parser.add_argument('--comm_round', type=int, default=100, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--datadir', type=str, required=False, default="../datasets/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--pretrain', default=False, type=bool, help='pretrain')    
    
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=2, help='the mu parameter for fedprox or moon')
    parser.add_argument('--out_dim', type=int, default=512, help='the output dimension for the projection layer')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--local_max_epoch', type=int, default=100, help='the number of epoch for local optimal training')
    parser.add_argument('--model_buffer_size', type=int, default=1, help='store how many previous models for contrastive loss')
    parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    parser.add_argument('--sample_fraction', type=float, default=1, help='how many clients are sampled in each round')
    parser.add_argument('--load_model_file', type=str, default=None, help='the model to load as global model')
    parser.add_argument('--load_pool_file', type=str, default=None, help='the old model pool path to load')
    parser.add_argument('--load_model_round', type=int, default=0, help='how many rounds have executed for the loaded model')
    parser.add_argument('--load_first_net', type=int, default=1, help='whether load the first net as old net or not')
    parser.add_argument('--normal_model', type=int, default=0, help='use normal model or aggregate model')
    parser.add_argument('--loss', type=str, default='contrastive')    
    parser.add_argument('--save_model',type=int,default=0)
    parser.add_argument('--use_project_head', type=int, default=1)
    parser.add_argument('--align', type=str, default="KL")
    parser.add_argument('--node_weight', type=float, default=1.0)
    parser.add_argument('--re_phase', type=str, default='p2')
    parser.add_argument('--edge_weight', type=float, default=1.0)
    parser.add_argument('--angle_weight', type=float, default=1.0)
    parser.add_argument('--dist_weight', type=float, default=1.0)
    parser.add_argument('--mode_dis', default='L2', type=str, help='The disloss mode')   
    parser.add_argument('--mode_angle', default='L2', type=str, help='The angleloss mode') 
    parser.add_argument('--re_bs', type=int, default=32)
    parser.add_argument('--re_k', type=int, default=2)
    parser.add_argument('--re_mu', type=float, default=1.0) 
    parser.add_argument('--shard_per_user', type=int, default=5) 
    
    args = parser.parse_args()
    return args

def get_updateModel_before(model, global_model):
    # 将global_model的参数更新到model上
    model_dict = model.state_dict()
    global_model_dict = global_model.state_dict()
  
    shared_dict = {k: v for k, v in global_model_dict.items() if k in model_dict}
#     import ipdb; ipdb.set_trace()   
    model_dict.update(shared_dict)
    
    model.load_state_dict(model_dict)
    return model


def get_updateModel_after(model, global_model):
    # 将model参数更新到global_model上
    model_dict = model.state_dict()
    global_model_dict = global_model.state_dict()
    
#     for k, v in dnn_dict.items():
#         if k in dnn_dict:
#             print(k)
#     print('********************************')   
#     import ipdb; ipdb.set_trace()    
    shared_dict = {k: v for k, v in model_dict.items() if k in global_model_dict}
#     import ipdb; ipdb.set_trace()  
    global_model_dict.update(shared_dict)
    
    
    global_model.load_state_dict(global_model_dict)
    return global_model


def find_missing_categories(data):
    # 假设类别的总数从0到9
    total_categories = set(range(10))

    # 创建一个空字典来存储每个客户端缺失的类别
    missing_categories = {}

    # 遍历每个客户端的数据
    for client_id, categories in data.items():
        # 找出该客户端缺失的类别
        missing = list(total_categories - set(categories.keys()))
        # 存储缺失的类别
        missing_categories[client_id] = missing

    return missing_categories



def init_nets(n_parties, args, n_classes, device='cuda:0'):
    nets = {net_i: None for net_i in range(n_parties)}
        
    clip_model, preprocess = clip.load("ViT-B/16", device=device)
    for net_i in range(n_parties):
        if 'cifar10' == args.dataset: 
            net = ModelFedCon(args.model, args.out_dim, n_classes)
            net.to(device)
            nets[net_i] = net
            Re_train_DNN = Re_Train_DNN_CIFAR10(84, [84, 512], n_classes)
        else:        
            if args.dataset == 'vireo172' or args.dataset == 'food101':
                net = resnet18(args.dataset, kernel_size=7, pretrained=args.pretrain)
                net.to(device)
                nets[net_i] = net
            else:
                net = resnet18(args.dataset, kernel_size=3, pretrained=args.pretrain)  
                net.to(device)
                nets[net_i] = net       
                Re_train_DNN = Re_Train_DNN_CIFAR10(512, [512, 512], n_classes)
    if 'cifar10' == args.dataset: 
        global_net = ModelFedCon(args.model, args.out_dim, n_classes)

    else:        
        if args.dataset == 'vireo172' or args.dataset == 'food101':
            global_net = resnet18(args.dataset, kernel_size=7, pretrained=args.pretrain)
        else:
            global_net = resnet18(args.dataset, kernel_size=3, pretrained=args.pretrain)  
    return nets, global_net, clip_model, Re_train_DNN



def train_net_fedavg(net_id, net, prototype, train_dataloader, epochs, lr, args_optimizer, args, device="cuda:0"):
    net.to(device)
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    prototype = prototype.to(device).float()
    label = np.array([range(prototype.shape[0])])
    
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss_ce_collector = []
        epoch_loss_align_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):

#             ipdb.set_trace()
            x, target = x.to(device), target.to(device)
            if args.dataset == 'pmnist':
                target = target.reshape(-1)
            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()
            _, feats, out = net(x)
#             ipdb.set_trace()
            loss_align = RKdNode(feats, target, prototype, t=args.temperature) 
#             ipdb.set_trace()
            loss_ce = criterion(out, target)
            loss = loss_ce + args.mu * loss_align

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss_ce_collector.append(loss_ce.item()) 
            epoch_loss_align_collector.append(loss_align.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss_ce = sum(epoch_loss_ce_collector) / len(epoch_loss_ce_collector)
        epoch_loss_align = sum(epoch_loss_align_collector) / len(epoch_loss_align_collector)
        logger.info('Epoch: %d Loss: %f Loss_ce: %f Loss_align: %f' % (epoch, epoch_loss, epoch_loss_ce, epoch_loss_align))
#     print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
#     print([p for name, p in net.named_parameters() if p.requires_grad==True and "fc" in name])
    net.to('cpu')
    logger.info(' ** Training complete **')
    return 0, 0



def local_train_net(nets, prototypes, args, net_dataidx_map, n_classes, missing=[], round=None, device="cuda:0"):
        
    local_proto_list = []
    local_proto_label_list = []
    local_vars_list = []
    k = 0
    for net_id, net in nets.items():
        dataidxs = net_dataidx_map[net_id]
        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, missing=missing[net_id])

        n_epoch = args.epochs
        trainacc, testacc = train_net_fedavg(net_id, net, prototypes, train_dl_local, n_epoch, args.lr,
                                              args.optimizer, args, device=device) 
        
        local_protos_i, local_labels_i, local_vars_i = dropout_proto_local_clustering(net, train_dl_local, args, n_class=n_classes)
        local_proto_list.append(local_protos_i)
        local_proto_label_list.append(local_labels_i)
        local_vars_list.append(local_vars_i)
        # ipdb.set_trace()
    return nets, torch.cat(local_proto_list), torch.cat(local_proto_label_list), torch.cat(local_vars_list)


if __name__ == '__main__':
    args = get_args()
    args.logdir = "./logs/+ISRC_CSPC/" +args.dataset+ "/"+args.re_phase
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    if args.log_file_name is None:
        argument_path = 'experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))
    log_path = args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.INFO, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(device)

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)


    logger.info("Partitioning data")
    X_train, y_train, X_btest, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)
    
#     X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = get_data(args)

    missing = find_missing_categories(traindata_cls_counts)
    
    n_party_per_round = int(args.n_parties * args.sample_fraction)
    party_list = [i for i in range(args.n_parties)]

    party_list_rounds = []
    if n_party_per_round != args.n_parties:
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)

    n_classes = len(np.unique(y_train))
    print(n_classes)
    train_dl_global, test_dl, train_ds_global, test_ds_global = get_dataloader(args.dataset, args.datadir,
                                                                               args.batch_size, 32)

    print("len train_dl_global:", len(train_ds_global))
    train_dl = None

    logger.info("Initializing nets")
    nets, global_model, clip_model, Re_train_DNN = init_nets(args.n_parties, args, n_classes, device=device)

    n_comm_rounds = args.comm_round
    name_class = name_classes(args)
    global_prototypes = generate_text_prototype(clip, clip_model, name_class, device=device).type(torch.float32)
    global_protos_labels = torch.tensor(list(range(n_classes)))
    print(global_prototypes.dtype)
#     ipdb.set_trace()
    for round in range(n_comm_rounds):
        logger.info("in comm round:" + str(round))
        party_list_this_round = party_list_rounds[round]
        global_w = global_model.state_dict()
        nets_this_round = {k: nets[k] for k in party_list_this_round}
        for net in nets_this_round.values():
            net.load_state_dict(global_w)

        nets_this_round, local_protos, local_proto_labels, local_vars = local_train_net(nets_this_round, global_prototypes, args, net_dataidx_map, n_classes, missing=missing, round=round, device=device)
        global_model.to('cpu')

        # update global model
        total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

        for net_id, net in enumerate(nets_this_round.values()):
            net.to('cpu')
            net_para = net.state_dict()
            if net_id == 0:
                for key in net_para:
                    global_w[key] = net_para[key] * fed_avg_freqs[net_id]
            else:
                for key in net_para:
                    global_w[key] += net_para[key] * fed_avg_freqs[net_id]
                    
        global_model.load_state_dict(global_w)

        logger.info('global n_training: %d' % len(train_dl_global))
        logger.info('global n_test: %d' % len(test_dl))

        if round >= 0:
            model = get_updateModel_before(Re_train_DNN, global_model)
            model = retrain_cls_relation_(model, local_protos, local_proto_labels, global_prototypes, n_classes, args, round, device, local_vars)   
            global_model.to(device)
            acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, args, get_confusion_matrix=True, device=device)
            logger.info('>> Global Before Model Test accuracy: %f' % acc)
            global_model = get_updateModel_after(model, global_model)        
        
        global_model.cuda()
        test_base_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, args, get_confusion_matrix=True, device=device)

        logger.info('>> Global Model Test accuracy: %f' % test_base_acc)


#         if round == 0 or round % 25 ==24:
#             mkdirs(args.modeldir + '+ISRC_CSPC/' + args.dataset + '/' + argument_path + '/' + str(round))
#             global_model.to('cpu')
#             torch.save(global_model.state_dict(),
#                        args.modeldir + '+ISRC_CSPC/' + args.dataset + '/' + argument_path + '/' + str(round) + '/global_model.pth')
#             for i in range(args.n_parties):
#                 torch.save(nets_this_round[i].state_dict(), args.modeldir + '+ISRC_CSPC/' + args.dataset + '/' + argument_path + '/' + str(round) + '/local_' + str(i) + '.pth')        
