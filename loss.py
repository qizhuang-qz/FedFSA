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
import math
import ipdb

import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss_text(nn.Module):
    def __init__(self, device="0", temperature=0.07, num_classes=10):
        super(SupConLoss_text, self).__init__()
        self.device = device
        self.temperature = temperature
        self.num_classes = num_classes

    def forward(self, features, labels, text_features):
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)  # [batch_size, 1]

        # 计算所有样本特征与所有类别文本特征的点积
        anchor_dot_contrast = torch.div(
            torch.matmul(features, text_features.T),
            self.temperature)  # [batch_size, num_classes]

        # 生成正例和负例掩码
        mask = torch.eq(labels, torch.arange(self.num_classes).to(self.device)).float()  # [batch_size, num_classes]

        # 用掩码标识正例和负例，进行 softmax 计算
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # [batch_size, num_classes]

        # 正例掩码贡献正损失，负例掩码贡献负损失
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob_pos.mean()

        return loss

    
    
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """Compute loss for model with 2D features.
        Args:
            features: hidden vectors of shape [batch_size, feature_dim].
            labels: ground truth of shape [batch_size].
        Returns:
            A loss scalar.
        """
        device = features.device
#         ipdb.set_trace()
        # Normalize the features
        features = F.normalize(features, dim=1)

        # Compute the similarity matrix
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # Create labels mask
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != features.shape[0]:
            raise ValueError('Number of labels does not match number of features')
        mask = torch.eq(labels, labels.T).float().to(device)

        # Compute log_prob
        exp_logits = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True))

        # Compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # Loss
        loss = -mean_log_prob_pos.mean()
        return loss
    
    

class SupervisedContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, x, y):
        # x: the feature representations of the samples
        # y: the ground truth labels

        # normalize the feature vectors
        x = F.normalize(x, dim=1)
#         ipdb.set_trace()
        # compute the similarity matrix
        sim_matrix = torch.matmul(x, x.t()) / self.temperature

        # generate the mask for positive and negative pairs
        mask = torch.eq(y.unsqueeze(0), y.unsqueeze(1)).float()
        mask = mask / mask.sum(dim=1, keepdim=True)

        # calculate the contrastive loss
        loss = (-torch.log_softmax(sim_matrix, dim=1) * mask).sum(dim=1).mean()

        return loss

def get_yh(y_de):
    yh = self.one_hot(y_de)
    return yh
    

    
    

class xERMLoss(nn.Module):
    def __init__(self, gamma, dataset='vireo'):
        super(xERMLoss, self).__init__()

        self.cls_loss = nn.CrossEntropyLoss(reduction='none')
        self.gamma = gamma
#         if critertion != None:
#             self.CE_loss = critertion

    def forward(self, logits_TE, logits_TDE, logits_student, labels):
        # calculate weight
        TDE_acc = self.cls_loss(logits_TDE, labels)
        TE_acc = self.cls_loss(logits_TE, labels)
        TDE_acc = torch.pow(TDE_acc, self.gamma)
        TE_acc = torch.pow(TE_acc, self.gamma)
        weight = TDE_acc/(TDE_acc + TE_acc)
        # student td loss
        te_loss = self.cls_loss(logits_student, labels)

        # student tde loss
        prob_tde = F.softmax(logits_TDE, -1).clone().detach()
        prob_student = F.softmax(logits_student, -1)
        tde_loss = - prob_tde * prob_student.log()
        tde_loss = tde_loss.sum(1)

        loss = (weight*tde_loss).mean() + ((1 - weight)*te_loss).mean()

        return loss


    
def nt_xent(x1, x2, t=0.07):
    """Contrastive loss objective function"""
    x1 = F.normalize(x1, dim=1)
    x2 = F.normalize(x2, dim=1)
    batch_size = x1.size(0)
    out = torch.cat([x1, x2], dim=0)
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / t)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
    pos_sim = torch.exp(torch.sum(x1 * x2, dim=-1) / t)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss

    
def PCLoss(features, f_labels, prototypes, p_labels, t=0.5):
    
    a_norm = features / features.norm(dim=1)[:, None]
    b_norm = prototypes / prototypes.norm(dim=1)[:, None]
    sim_matrix = torch.exp(torch.mm(a_norm, b_norm.transpose(0,1)) / t)
    
    pos_sim = torch.exp(torch.diag(torch.mm(a_norm, b_norm[f_labels].transpose(0,1))) / t)
    
    loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    
    return loss

def refine_as_not_true(logits, targets, num_classes):
    nt_positions = torch.arange(0, num_classes).to(logits.device)
    nt_positions = nt_positions.repeat(logits.size(0), 1)
    nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]
    nt_positions = nt_positions.view(-1, num_classes - 1)

    logits = torch.gather(logits, 1, nt_positions)

    return logits

class NTD_Loss(nn.Module):
    """Not-true Distillation Loss"""

    def __init__(self, num_classes=10, tau=3, lamb=1):
        super(NTD_Loss, self).__init__()
        self.CE = nn.CrossEntropyLoss()
        self.MSE = nn.MSELoss()
        self.KLDiv = nn.KLDivLoss(reduction="batchmean")
        self.num_classes = num_classes
        self.tau = tau
        self.beta = lamb

    def forward(self, logits, targets, dg_logits):
        ce_loss = self.CE(logits, targets)
        ntd_loss = self._ntd_loss(logits, dg_logits, targets)

        loss = ce_loss + self.beta * ntd_loss

        return loss

    def _ntd_loss(self, logits, dg_logits, targets):
        """Not-tue Distillation Loss"""

        # Get smoothed local model prediction
        logits = refine_as_not_true(logits, targets, self.num_classes)
        pred_probs = F.log_softmax(logits / self.tau, dim=1)

        # Get smoothed global model prediction
        with torch.no_grad():
            dg_logits = refine_as_not_true(dg_logits, targets, self.num_classes)
            dg_probs = torch.softmax(dg_logits / self.tau, dim=1)

        loss = (self.tau ** 2) * self.KLDiv(pred_probs, dg_probs)

        return loss

class WeightedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, target, weights):
        # logits: [N, C, H, W], target: [N, H, W]
        # loss = sum(-y_i * log(c_i))

        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1), -1)  # [N, C, HW]
            logits = logits.transpose(1, 2)  # [N, HW, C]
            logits = logits.contiguous().view(-1, logits.size(2))  # [NHW, C]
        target = target.view(-1, 1)  # [NHW，1]

        logits = F.log_softmax(logits, 1)

#         import ipdb; ipdb.set_trace()

        logits = logits.gather(1, target).reshape(-1)  # [NHW, 1]

        loss = -1 * (torch.mul(logits, weights))

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return torch.sum(loss, 0)
        
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return (1 - lam) * criterion(pred, y_a) + lam * criterion(pred, y_b)
    

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        dist_pos = torch.norm(anchor - positive, 2, dim=1)
        dist_neg = torch.norm(anchor - negative, 2, dim=1)
        loss = torch.mean(torch.clamp(dist_pos - dist_neg + self.margin, min=0.0))
        return loss


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res    


class KDLoss(nn.Module):
    '''
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    '''
    def __init__(self, T):
        super(KDLoss, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        # kd = F.kl_div(F.log_softmax(out_s / self.T, dim=1),
        #               F.softmax(out_t / self.T, dim=1),
        #               reduction='none').mean(dim=0)
        kd_loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
                        F.softmax(out_t/self.T, dim=1),
                        reduction='batchmean') * self.T * self.T

        return kd_loss

def js_divergence(p, q):
    kl_loss = KDLoss(T=0.5).cuda()
    #     ipdb.set_trace()
    half = torch.div(p + q, 2)
    s1 = kl_loss(p, half)
    s2 = kl_loss(q, half)
    #     ipdb.set_trace()
    return torch.div(s1 + s2, 2)    
    
    
class RKdAngle(nn.Module):
    def forward(self, student, teacher, args):
        # N x C
        # N x N x C
        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1)) + 1e-6
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)
#         import ipdb; ipdb.set_trace()
        if args.mode_angle == 'L2':
            loss = F.mse_loss(s_angle, t_angle, reduction='mean')
        elif args.mode_angle == 'L1':
            loss = F.l1_loss(s_angle, t_angle, reduction='mean')
        elif args.mode_angle == 'smooth_l1':          
            loss = F.smooth_l1_loss(s_angle, t_angle, reduction='mean')
#         if (torch.isnan(loss).any()):
#             ipdb.set_trace()
        return loss


class RkdDistance(nn.Module):
    def forward(self, student, teacher, args):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td
#         ipdb.set_trace()
        d = pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d
        if args.mode_dis == 'KL':
            KL_loss = KDLoss(T=0.5).to(args.device)   
            loss = KL_loss(d, t_d)
        elif args.mode_dis == 'L2':
            loss = F.mse_loss(d, t_d, reduction='mean')
        elif args.mode_dis == 'L1':
            loss = F.l1_loss(d, t_d, reduction='mean')
        elif args.mode_dis == 'smooth_l1':
            loss = F.smooth_l1_loss(d, t_d, reduction='mean')            
        elif args.mode_dis == 'JS':
            loss = js_divergence(d, t_d)    
#         if (torch.isnan(loss).any()):
#             ipdb.set_trace()            
        return loss

    
class Relation_Loss(nn.Module):
    def __init__(self, dist_weight=1.0, angle_weight=1.0):
        super(Relation_Loss, self).__init__()
        self.dist_criterion = RkdDistance()
        self.angle_criterion = RKdAngle()
        self.dist_weight = dist_weight
        self.angle_weight = angle_weight       
    
    def forward(self, student, teacher, args):                

        dis_loss = self.dist_criterion(student, teacher, args)
        angle_loss = self.angle_criterion(student, teacher, args)
        relational_loss = self.dist_weight * dis_loss + self.angle_weight * angle_loss
#         import ipdb; ipdb.set_trace()
        return relational_loss   
    
    
    
class Relation_Local_Loss(nn.Module):
    def __init__(self, dist_weight=1, angle_weight=1, mode='mean'):
        super(Relation_Local_Loss, self).__init__()
        self.dist_criterion = RkdDistance()
        self.angle_criterion = RKdAngle()
        self.dist_weight = dist_weight
        self.angle_weight = angle_weight 
        self.mode = mode
    
    def forward(self, student, teacher, client_ids, args):    
        
        unique_client, counts = torch.unique(client_ids, return_counts=True)
        relational_loss = 0
        n = 0
        for k in unique_client:
            indices = torch.where(client_ids == k)[0]
            if len(indices) >= 2:
                dis_loss = self.dist_criterion(student[indices], teacher[indices], args)
                angle_loss = self.angle_criterion(student[indices], teacher[indices], args)
        
                relational_loss += self.dist_weight * dis_loss + self.angle_weight * angle_loss
                n += 1
        if self.mode == 'sum':
            return relational_loss
        elif self.mode == 'mean':
            return relational_loss / n
        
    
class Relation_Complementary_Loss(nn.Module):
    def __init__(self, dist_weight=1, angle_weight=1):
        super(Relation_Complementary_Loss, self).__init__()
        self.dist_criterion = RkdDistance()
        self.angle_criterion = RKdAngle()
        self.dist_weight = dist_weight
        self.angle_weight = angle_weight       
    
    def forward(self, student, teacher, args):    
        dis_loss = self.dist_criterion(student, teacher, args)
        angle_loss = self.angle_criterion(student, teacher, args)
        relational_loss = self.dist_weight * dis_loss + self.angle_weight * angle_loss
        
        return relational_loss    
    

class Local_Edge_Loss(nn.Module):
    def __init__(self, mode='sum'):
        super(Local_Edge_Loss, self).__init__()
        self.dist_criterion = RkdDistance()
        self.mode = mode
    
    def forward(self, student, teacher, client_ids, args):    
        
        unique_client, counts = torch.unique(client_ids, return_counts=True)
        relational_loss = 0
        n = 0
        for k in unique_client:
            indices = torch.where(client_ids == k)[0]
            if len(indices) >= 3:
                dis_loss = self.dist_criterion(student[indices], teacher[indices], args)        
                relational_loss += dis_loss
                n += 1
        if self.mode == 'sum':
            return relational_loss
        elif self.mode == 'mean':
            return relational_loss / n    
    
class Local_Angle_Loss(nn.Module):
    def __init__(self, mode='sum'):
        super(Local_Angle_Loss, self).__init__()
        self.angle_criterion = RKdAngle()
        self.mode = mode
    
    def forward(self, student, teacher, client_ids, args):            
        unique_client, counts = torch.unique(client_ids, return_counts=True)
        relational_loss = 0
        n = 0
        for k in unique_client:
            indices = torch.where(client_ids == k)[0]
            if len(indices) >= 3:
                dis_loss = self.angle_criterion(student[indices], teacher[indices], args)        
                relational_loss += dis_loss
                n += 1
        if self.mode == 'sum':
            return relational_loss
        elif self.mode == 'mean':
            return relational_loss / n     
        
class Com_Edge_Loss(nn.Module):
    def __init__(self):
        super(Com_Edge_Loss, self).__init__()
        self.dist_criterion = RkdDistance()
    
    def forward(self, student, teacher, args):    
        dis_loss = self.dist_criterion(student, teacher, args)        
        relational_loss = dis_loss

        return relational_loss
  
    
class Com_Angle_Loss(nn.Module):
    def __init__(self):
        super(Com_Angle_Loss, self).__init__()
        self.angle_criterion = RKdAngle()
    
    def forward(self, student, teacher, args):            
        dis_loss = self.angle_criterion(student, teacher, args)        
        relational_loss = dis_loss

        return relational_loss         
        
class ArcFace(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, s=64.0, margin=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False


    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()
            final_target_logit = target_logit + self.margin
            logits[index, labels[index].view(-1)] = final_target_logit
            logits.cos_()
        logits = logits * self.s   
        return logits

def PCLoss(features, f_labels, prototypes, p_labels, t=0.5):
    
    a_norm = features / features.norm(dim=1)[:, None]
    b_norm = prototypes / prototypes.norm(dim=1)[:, None]
    sim_matrix = torch.exp(torch.mm(a_norm, b_norm.transpose(0,1)) / t)
    
    pos_sim = torch.exp(torch.diag(torch.mm(a_norm, b_norm[f_labels].transpose(0,1))) / t)
    
    loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    
    return loss    
    
def balanced_softmax_loss(logits, labels, sample_per_class, reduction="mean"):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss    



# def RKdNode(features, f_labels, prototypes, label, t=0.5):

#     unique_classes = np.array(label)
#     selected_prototypes = prototypes[unique_classes]
#     features = features / torch.norm(features, dim=1, keepdim=True)
#     selected_prototypes = selected_prototypes / torch.norm(selected_prototypes, dim=1, keepdim=True)
#     sim_matrix = torch.exp(torch.mm(features, selected_prototypes.transpose(0, 1)) / t)
#     pos_sim = torch.exp(torch.diag(torch.mm(features, selected_prototypes[f_labels].transpose(0, 1))) / t)
#     loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

#     return loss

def RKdNode(features, f_labels, prototypes, t=0.5):
    """
    Compute the loss based on the similarity between features and the prototypes 
    corresponding to the unique classes in f_labels.

    :param features: Feature matrix from the model, shape (batch_size, feature_dim)
    :param f_labels: Labels for the batch, shape (batch_size,)
    :param prototypes: Prototypes for each class, shape (num_classes, feature_dim)
    :param t: Temperature parameter for scaling
    :return: Computed loss
    """   
    # Normalize features and prototypes
    features = features / torch.norm(features, dim=1, keepdim=True)
    prototypes = prototypes / torch.norm(prototypes, dim=1, keepdim=True)
    
    # Compute similarity matrix for all classes
    sim_matrix = torch.exp(torch.mm(features, prototypes.transpose(0, 1)) / t)
    
    # Compute positive similarity
    pos_sim = torch.exp(torch.sum(features * prototypes[f_labels], dim=1) / t)
    
    # Calculate loss
    loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

    return loss

