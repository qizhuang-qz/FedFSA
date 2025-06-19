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
from torch.utils.data import ConcatDataset
from torchvision import datasets
from sampling import *
from datasets import *
from kmeans import *
import ipdb
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def log_model_info(model, verbose=False):
    """Logs model info"""
    if verbose:
        logger.info(f"Classification Model:\n{model}")
    model_total_params = sum(p.numel() for p in model.parameters())

    model_grad_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total Parameters: {0}\t Gradient Parameters: {1}".format(
        model_total_params, model_grad_params))
    logger.info("tuned percent:%.3f"%(model_grad_params/model_total_params*100))    
    

def load_cifar10_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform_local=transform, transform_clip=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform_local=transform, transform_clip=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def load_cifar100_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform_local=transform, transform_clip=transform)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform_local=transform, transform_clip=transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)




def load_tinyimagenet_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])
    xray_train_ds = TinyImageNet_load(datadir+'tiny-imagenet-200/', train=True, transform_local=transform, transform_clip=transform)
    xray_test_ds = TinyImageNet_load(datadir+'tiny-imagenet-200/', train=False, transform_local=transform, transform_clip=transform)

    X_train, y_train = np.array([s[0] for s in xray_train_ds.samples]), np.array([int(s[1]) for s in xray_train_ds.samples])
    X_test, y_test = np.array([s[0] for s in xray_test_ds.samples]), np.array([int(s[1]) for s in xray_test_ds.samples])

    return (X_train, y_train, X_test, y_test)

def load_vireo_data():
    transform = transforms.Compose([transforms.ToTensor()])

    vireo_train_ds = Vireo172_truncated(transform=transform, mode='train')
    vireo_test_ds = Vireo172_truncated(transform=transform, mode='test')

    X_train, y_train = vireo_train_ds.path_to_images, vireo_train_ds.labels
    X_test, y_test = vireo_test_ds.path_to_images, vireo_test_ds.labels

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)

def load_food_data():
    transform = transforms.Compose([transforms.ToTensor()])

    vireo_train_ds = Food101_truncated(transform=transform, mode='train')
    vireo_test_ds = Food101_truncated(transform=transform, mode='test')

    X_train, y_train = vireo_train_ds.path_to_images, vireo_train_ds.labels
    X_test, y_test = vireo_test_ds.path_to_images, vireo_test_ds.labels

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)




def record_net_data_stats(y_train, net_dataidx_map, logdir):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    data_list=[]
    for net_id, data in net_cls_counts.items():
        n_total=0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))
    logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts


def partition_data(dataset, datadir, logdir, partition, n_parties, beta=0.4):
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    elif dataset == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist_data(datadir)
    elif dataset == 'fmnist':
        X_train, y_train, X_test, y_test = load_fmnist_data(datadir)        
        
    elif dataset == 'cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
    elif dataset == 'tinyimagenet':
        X_train, y_train, X_test, y_test = load_tinyimagenet_data(datadir)
    elif dataset == 'vireo172':
        X_train, y_train, X_test, y_test = load_vireo_data()
    elif dataset == 'food101':
        X_train, y_train, X_test, y_test = load_food_data()        
        
    n_train = y_train.shape[0]

    if partition == "homo" or partition == "iid":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}


    elif partition == "noniid-labeldir" or partition == "noniid":
        min_size = 0
        min_size_test = 0
        min_require_size = 10
        K = 10
        if dataset == 'cifar100':
            K = 100
        elif dataset == 'tinyimagenet':
            K = 200
            # min_require_size = 100
        elif dataset == 'vireo172':
            K = 172
        elif dataset == 'food101':
            K = 101    
            
        N_train = y_train.shape[0]
        N_test = y_test.shape[0]
        
        net_dataidx_map = {}
        net_dataidx_map_test = {}
        while min_size < min_require_size and min_size_test < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            idx_batch_test = [[] for _ in range(n_parties)]
            for k in range(K):
                
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array([p * (len(idx_j) < N_train / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])


        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]           

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    
    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)

def compute_accuracy_text(model, dataloader, prototypes, get_confusion_matrix=False, device="cpu", multiloader=False):
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])
    model.to(device)
    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            if device != 'cpu':
                x, target = x.to(device), target.to(dtype=torch.int64).to(device)
   
            logits_per_image = model(x)  
                
            out = logits_per_image.softmax(dim=-1).float()    
            _, pred_label = torch.max(out.data, 1)

            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()
#             print(target)
#             print(out)
            
            if device == "cpu":
                pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                true_labels_list = np.append(true_labels_list, target.data.numpy())                             
            else:
                pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    avg_loss = 0    
    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        return correct / float(total), conf_matrix, avg_loss

    return correct / float(total), avg_loss


def compute_accuracy(global_model, dataloader, args, get_confusion_matrix=True, device="cuda:0"):
    true_labels_list, pred_labels_list = np.array([]), np.array([])
    correct, total = 0, 0

    criterion = nn.CrossEntropyLoss().cuda()
    loss_collector = []
    global_model.eval()
    
    global_model = global_model.cuda()
    
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            #print("x:",x)
            if args.dataset == 'pmnist':
                target = target.reshape(-1)
            if device != 'cpu':
                x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            target = target.long()
#             feats = global_model.encode_image(x.float())
            _,_,out = global_model(x.float())
#             ipdb.set_trace()
            loss = criterion(out, target)
            _, pred_label = torch.max(out.data, 1)
            loss_collector.append(loss.item())
            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()

            if device == "cpu":
                pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                true_labels_list = np.append(true_labels_list, target.data.numpy())
            else:
                pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
        avg_loss = sum(loss_collector) / len(loss_collector)

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if get_confusion_matrix:
        return correct / float(total), conf_matrix, avg_loss

    return correct / float(total), avg_loss



def get_data(args, missing=[]):
    if args.dataset == 'cifar10':
        trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
        trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
        dataset_train = datasets.CIFAR10('../datasets/', train=True, download=True, transform=trans_cifar10_train)
        dataset_test = datasets.CIFAR10('../datasets/', train=False, download=True, transform=trans_cifar10_val)
        
        X_train, y_train = dataset_train.data, np.array(dataset_train.targets)
        X_test, y_test = dataset_test.data, np.array(dataset_test.targets)
        
        if args.partition == 'iid':
            dict_users_train = iid(dataset_train, args.n_parties)
            dict_users_test = iid(dataset_test, args.n_parties)
        else:
            dict_users_train, rand_set_all = noniid(dataset_train, y_train, args.n_parties, args.shard_per_user)

    elif args.dataset == 'cifar100':
        trans_cifar100_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                               std=[0.267, 0.256, 0.276])])
        trans_cifar100_val = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                              std=[0.267, 0.256, 0.276])])
        dataset_train = datasets.CIFAR100('../datasets/', train=True, download=True, transform=trans_cifar100_train)
        dataset_test = datasets.CIFAR100('../datasets/', train=False, download=True, transform=trans_cifar100_val)
        
        X_train, y_train = dataset_train.data, np.array(dataset_train.targets)
#         X_train, y_train = dataset_train.data, torch.tensor(dataset_train.targets)
        X_test, y_test = dataset_test.data, np.array(dataset_test.targets)
        
        if args.partition == 'iid':
            dict_users_train = iid(dataset_train, args.n_parties)
            dict_users_test = iid(dataset_test, args.n_parties)
        else:
            dict_users_train, rand_set_all = noniid(dataset_train, y_train, args.n_parties, args.shard_per_user)
#             dict_users_test, rand_set_all = noniid(dataset_test, args.n_parties, args.shard_per_user, rand_set_all=rand_set_all)
    elif args.dataset == 'fmnist':
        trans_mnist = transforms.Compose([transforms.ToTensor()])
        dataset_train = datasets.FashionMNIST('data/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.FashionMNIST('data/', train=False, download=True, transform=trans_mnist)

        X_train, y_train = dataset_train.train_data, dataset_train.train_labels
        print(y_train)
        X_test, y_test = dataset_test.test_data, dataset_test.test_labels

        # sample users
        if args.partition == 'iid':
            dict_users_train = iid(dataset_train, args.n_parties)
            dict_users_test = iid(dataset_test, args.n_parties)
        else:
            dict_users_train, rand_set_all = noniid(dataset_train, args.n_parties, args.shard_per_user)
#             print('dict_users_train', dict_users_train)
            dict_users_test, rand_set_all = noniid(dataset_test, args.n_parties, args.shard_per_user, rand_set_all=rand_set_all)
   
    
    elif args.dataset == 'tinyimagenet':
        dl_obj = TinyImageNet_load
        transform_train = transforms.Compose([
            
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        dataset_train = dl_obj('../datasets/tiny-imagenet-200/', transform=transform_train)
        dataset_test = dl_obj('../datasets/tiny-imagenet-200/', transform=transform_test)

        X_train, y_train = np.array(dataset_train.samples)[:][0], np.array(dataset_train.samples)[:,1]
#         X_train, y_train = dataset_train.data, torch.tensor(dataset_train.targets)
        X_test, y_test = np.array(dataset_test.samples)[:][0], np.array(dataset_test.samples)[:,1]
        
        if args.partition == 'iid':
            dict_users_train = iid(dataset_train, args.n_parties)
            dict_users_test = iid(dataset_test, args.n_parties)
        else:
            dict_users_train, rand_set_all = noniid(dataset_train, y_train, args.n_parties, args.shard_per_user)
#             dict_users_test, rand_set_all = noniid(dataset_test, args.n_parties, args.shard_per_user, rand_set_all=rand_set_all)
    
    
    else:
        exit('Error: unrecognized dataset')
    traindata_cls_counts = record_net_data_stats(y_train, dict_users_train, args.logdir)
    return X_train, y_train, X_test, y_test, dict_users_train, traindata_cls_counts





def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, missing=[], noise_level=0, drop_last=True):
    if dataset in ('cifar10', 'cifar100'):
        if dataset == 'cifar10':
            dl_obj = CIFAR10_truncated

            normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        
            
            transform_train_clip = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.Resize(224),
                transforms.ColorJitter(brightness=noise_level),
#                 transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
            
            transform_train_local = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=noise_level),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
            
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize])

        elif dataset == 'cifar100':
            dl_obj = CIFAR100_truncated

            normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])

            transform_train_clip = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.Resize(224),
                transforms.ColorJitter(brightness=noise_level),
#                 transforms.RandomCrop(32),
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
            transform_train_local = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=noise_level),
                transforms.RandomCrop(32),
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize_local
            ])

            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize])
        
        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform_local=transform_train_local, transform_clip=transform_train_clip, download=True)
        test_ds = dl_obj(datadir, train=False, transform_local=transform_test, transform_clip=transform_train_local, download=True)
        
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=drop_last, shuffle=True, num_workers=4, pin_memory=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=4, pin_memory=True)

    elif dataset == 'tinyimagenet':
        dl_obj = TinyImageNet_load
        transform_train_clip = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        transform_train_local = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(
                Variable(x.unsqueeze(0), requires_grad=False),
                (4, 4, 4, 4), mode='reflect').data.squeeze()),
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=noise_level),
            transforms.RandomCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])        
        
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


        train_ds = dl_obj('../datasets/tiny-imagenet-200/', train=True, dataidxs=dataidxs, transform_local=transform_train_local, transform_clip=transform_train_clip)
        test_ds = dl_obj('../datasets/tiny-imagenet-200/', train=False, transform_local=transform_test, transform_clip=transform_train_clip)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=drop_last, shuffle=True, num_workers=4, pin_memory=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=4, pin_memory=True)

    elif dataset == 'vireo172':
        dl_obj = Vireo172_truncated
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.Resize(224),
                transforms.ColorJitter(brightness=noise_level),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
#         transform_train = transforms.Compose([
#         transforms.ToTensor(),
#         normalize,
#          ])       
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize,
         ])

        
        train_ds = dl_obj(dataidxs, transform_train, mode='train')
        test_ds = dl_obj(None, transform_test, mode='test')
        
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=drop_last, shuffle=True, num_workers=8, pin_memory=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=8, pin_memory=True)          
        
    elif dataset == 'food101':
        dl_obj = Food101_truncated
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.Resize(224),
                transforms.ColorJitter(brightness=noise_level),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
#         transform_train = transforms.Compose([
#         transforms.ToTensor(),
#         normalize,
#          ])       
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize,
         ])

        
        train_ds = dl_obj(dataidxs, transform_train, mode='train')
        test_ds = dl_obj(None, transform_test, mode='test')
        
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=drop_last, shuffle=True, num_workers=2, pin_memory=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=2, pin_memory=True)          
        
    return train_dl, test_dl, train_ds, test_ds





def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval() 

def column_variance(arr):
    # 计算每一列的方差，沿着第0维计算
    column_vars = np.var(arr, axis=0, ddof=0)
    return column_vars        
        
        
# p2        
def dropout_proto_local_clustering(net, dataloader, args, n_class=10, device='cuda:0'):
    feats = []
    labels = []
    net.eval()
    net.apply(fix_bn)
    net.to('cpu')
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            x, target = x.to('cpu'), target.to('cpu')
            feat, _, _ = net(x)
            
            if batch_idx == 0:
                feats = feat
                labels = target
            else:
                feats = torch.cat([feats, feat])
                labels = torch.cat([labels, target])

    prototype = []
    proto_label = []
    var = []
    
    if args.dataset == 'cifar10':
        k, cs = 50, 15
    elif args.dataset == 'cifar100' or args.dataset == 'tinyimagenet':
        k, cs = 40, 10            
    

    feats = feats.numpy()
    labels = labels.numpy()
    kmeans = KMEANS(n_clusters=1, max_iter=20)
    cluster_size_max = 0
    for i in range(n_class):
        idx_i = np.where(labels == i)[0]
        if len(idx_i) >= k:
            predict_labels_i = kmeans.fit(feats[idx_i])
            cluster_i_set, unq_cluster_i_size = np.unique(predict_labels_i, return_counts=True)
            
            for cluster_id, cluster_size in zip(cluster_i_set, unq_cluster_i_size): 
                data_cluster_id = np.where(predict_labels_i == cluster_id)[0]
                clusters_var = column_variance(feats[idx_i][data_cluster_id])
                if cluster_size > cluster_size_max:
                    cluster_size_max = cluster_size
                    max_var = clusters_var
                                
                feature_classwise = feats[idx_i][data_cluster_id]
                clusters_var = column_variance(feats[idx_i][data_cluster_id])
                proto, average_distance_to_proto = compute_cluster_center_and_average_distance(feature_classwise)   
                prototype.append(proto)                        
                proto_label.append(int(i))
                var.append(clusters_var)
                                        
        elif len(idx_i) > 0 and len(idx_i) < k:  
            clusters_var = column_variance(feats[idx_i])   
            
            feature_classwise = feats[idx_i]
            proto, average_distance_to_proto = compute_cluster_center_and_average_distance(feature_classwise)    
            prototype.append(proto)                        
            proto_label.append(int(i))
            var.append(clusters_var)    
                               
#     ipdb.set_trace()
    prototype = np.vstack(prototype)
    proto_label = np.array(proto_label) 

    var = 0.9 * np.vstack(var) + 0.1 * max_var
           
    return torch.tensor(prototype).to(device), torch.tensor(proto_label).to(device), torch.tensor(var).to(device)         
        

def dropout_proto_local_final(net, dataloader, args, n_class=10, device='cuda:0'):
    feats = []
    labels = []
    net.eval()
    net.apply(fix_bn)
    net.to('cpu')
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            x, target = x.to('cpu'), target.to('cpu')
            feat, _, _ = net(x)
            
            if batch_idx == 0:
                feats = feat
                labels = target
            else:
                feats = torch.cat([feats, feat])
                labels = torch.cat([labels, target])

    prototype = []
    proto_label = []
    
    if args.dataset == 'cifar10':
        k, cs = 50, 15
    elif args.dataset == 'cifar100' or args.dataset == 'tinyimagenet':
        k, cs = 40, 10            
    
    CS = []
    ADtP = []
#     ipdb.set_trace()
    feats = feats.numpy()
    labels = labels.numpy()
    kmeans = KMEANS(n_clusters=2, max_iter=20)
    
    cluster_size_max = 0
    indicators = []
    for i in range(n_class):
        idx_i = np.where(labels == i)[0]
        if len(idx_i) >= k:
            predict_labels_i = kmeans.fit(feats[idx_i])
            cluster_i_set, unq_cluster_i_size = np.unique(predict_labels_i, return_counts=True)
            
            for cluster_id, cluster_size in zip(cluster_i_set, unq_cluster_i_size): 
                data_cluster_id = np.where(predict_labels_i == cluster_id)[0]
                if cluster_size >= cs:
                    if cluster_size > cluster_size_max:
                        max_clusters_var = column_variance(feats[idx_i][data_cluster_id])
                        cluster_size_max = cluster_size
                    for j in range(3): #len(class_idx[i])
                        idx = np.random.choice(np.arange(cluster_size), int(cluster_size*0.6))  
                        feature_classwise = feats[idx_i][data_cluster_id[idx]]
                        proto, average_distance_to_proto = compute_cluster_center_and_average_distance(feature_classwise)   
                        prototype.append(proto)                        
                        proto_label.append(int(i))
                        CS.append(cluster_size)
                        ADtP.append(average_distance_to_proto)
                        indicators.append(-1)
                else:
                    feature_classwise = feats[idx_i][data_cluster_id]
                    proto, average_distance_to_proto = compute_cluster_center_and_average_distance(feature_classwise)   
                    prototype.append(proto)                        
                    proto_label.append(int(i))
                    CS.append(cluster_size)
                    ADtP.append(average_distance_to_proto)  
                    indicators.append(0)
                                        
        elif len(idx_i) > cs and len(idx_i) < k:
            print(len(idx_i))
            for j in range(3):
                idx = np.random.choice(np.arange(len(idx_i)), int(len(idx_i)*0.6))                   
                feature_classwise = feats[idx_i[idx]]
                proto, average_distance_to_proto = compute_cluster_center_and_average_distance(feature_classwise)    
                prototype.append(proto)                        
                proto_label.append(int(i))
                CS.append(len(idx_i))
                ADtP.append(average_distance_to_proto) 
                indicators.append(0)
                
        elif len(idx_i) > 0 and len(idx_i) <= cs:   
            print(len(idx_i))
            feature_classwise = feats[idx_i]
            proto, average_distance_to_proto = compute_cluster_center_and_average_distance(feature_classwise)    
            prototype.append(proto)                        
            proto_label.append(int(i))
            CS.append(len(idx_i))
            ADtP.append(average_distance_to_proto)
            indicators.append(0)
                               
#     ipdb.set_trace()
    prototype = np.vstack(prototype)
    proto_label = np.array(proto_label) 
    CS = np.array(CS) / np.mean(CS)  
    ADtP = np.array(ADtP) / np.mean(ADtP)  
#     ipdb.set_trace()
    min_distances_to_other_classes, min_indices_to_other_classes = compute_min_distances(prototype, proto_label)
    min_distances_to_other_classes = min_distances_to_other_classes / np.mean(min_distances_to_other_classes) 
    representative_score = CS * min_distances_to_other_classes / ADtP
    representative_score = truncate_array(representative_score)
           
    return torch.tensor(prototype).to(device), torch.tensor(proto_label).to(device), representative_score, torch.tensor(max_clusters_var), indicators  


def compute_cluster_center_and_average_distance(cluster_features):
    """
    计算簇中心和簇内样本的平均距离。
    
    参数：
    cluster_features (np.ndarray): 包含簇内特征的NumPy数组，每行代表一个样本，每列代表一个特征。
    
    返回：
    cluster_center (np.ndarray): 簇中心，即特征的平均值。
    average_distance_to_center (float): 簇内样本的平均距离。
    """
    if cluster_features.shape[0] > 1:
    # 计算簇中心：计算特征的平均值
        cluster_center = np.mean(cluster_features, axis=0)

        # 计算簇内样本的平均距离：计算每个样本与簇中心的距离，并取平均值
        distances_to_center = np.linalg.norm(cluster_features - cluster_center, axis=1)
        average_distance_to_center = np.mean(distances_to_center)
    else:
        cluster_center = cluster_features
        average_distance_to_center = 1e-5

    return cluster_center, average_distance_to_center


def compute_min_distances(prototypes, class_labels):
    num_prototypes = prototypes.shape[0]
    num_classes = len(np.unique(class_labels))
    
    # 使用广播计算每对原型之间的欧几里得距离
    distances = np.linalg.norm(prototypes[:, np.newaxis] - prototypes, axis=2)

    # 将对角线上的距离设置为无穷大，避免自身与自身的距离影响结果
    distances += np.eye(num_prototypes) * (np.max(distances) + 1.0)

    min_distances_to_other_classes = np.zeros(num_prototypes)
    min_indices_to_other_classes = np.zeros(num_prototypes, dtype=np.int)

    for i in range(num_prototypes):
        same_class_indices = (np.array(class_labels) == class_labels[i])
        different_class_indices = ~same_class_indices

        min_distance = np.min(distances[i, different_class_indices])
        min_distance_index = np.argmin(distances[i, different_class_indices])

        min_distances_to_other_classes[i] = min_distance
        min_indices_to_other_classes[i] = min_distance_index.item()

    return min_distances_to_other_classes, min_indices_to_other_classes


def mixup_data(x, y, alpha=0.2):
    lam = torch.distributions.beta.Beta(alpha, alpha).sample().item()
    mixed_f = lam * x + (1 - lam) * y

    return mixed_f

