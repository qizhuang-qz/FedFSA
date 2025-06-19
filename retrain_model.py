import torch
import torch.nn as nn
import torch.nn.functional as F
import math

    
class Causal_Norm_Classifier(nn.Module):
    
    def __init__(self, num_classes=172, feat_dim=84, use_effect=True, num_head=2, tau=16.0, alpha=0.15, gamma=0.03125, *args):
        super(Causal_Norm_Classifier, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_classes, feat_dim).cuda(), requires_grad=True)
        self.scale = tau / num_head   # 16.0 / num_head
        self.norm_scale = gamma       # 1.0 / 32.0
        self.alpha = alpha            # 3.0
        self.num_head = num_head
        self.head_dim = feat_dim // num_head
        self.use_effect = use_effect
        self.reset_parameters(self.weight)
        self.relu = nn.ReLU(inplace=True)
        
    def reset_parameters(self, weight):
        stdv = 1. / math.sqrt(weight.size(1))
        weight.data.uniform_(-stdv, stdv)
    
    def forward(self, x, embed):
        # calculate capsule normalized feature vector and predict
        
        normed_w = self.multi_head_call(self.causal_norm, self.weight, weight=self.norm_scale)
        normed_x = self.multi_head_call(self.l2_norm, x)
        y_TE = torch.mm(normed_x * self.scale, normed_w.t())
        y_TDE = y_TE.clone()

        # remove the effect of confounder c during test
        if (not self.training) and self.use_effect:
            self.embed = torch.from_numpy(embed).view(1, -1).to(x.device)
            normed_c = self.multi_head_call(self.l2_norm, self.embed)
            head_dim = x.shape[1] // self.num_head
            x_list = torch.split(normed_x, head_dim, dim=1)
            c_list = torch.split(normed_c, head_dim, dim=1)
            w_list = torch.split(normed_w, head_dim, dim=1)
            output = []

            for nx, nc, nw in zip(x_list, c_list, w_list):
                cos_val, sin_val = self.get_cos_sin(nx, nc)
                y0 = torch.mm((nx -  cos_val * self.alpha * nc) * self.scale, nw.t())
                output.append(y0)
            y_TDE = sum(output)
            
        return y_TDE, y_TE
    
    def forward_old(self, x, embed):
        # calculate capsule normalized feature vector and predict
        normed_w = self.multi_head_call(self.causal_norm, self.weight, weight=self.norm_scale)
        normed_x = self.multi_head_call(self.l2_norm, x)
        y_TE = torch.mm(normed_x * self.scale, normed_w.t())
        y_TDE = y_TE.clone()
        return normed_x*self.scale, normed_w,y_TDE, y_TE

        # remove the effect of confounder c during test
        if (not self.training) and self.use_effect:
            self.embed = torch.from_numpy(embed).view(1, -1).to(x.device)
            normed_c = self.multi_head_call(self.l2_norm, self.embed)
            head_dim = x.shape[1] // self.num_head
            x_list = torch.split(normed_x, head_dim, dim=1)
            c_list = torch.split(normed_c, head_dim, dim=1)
            w_list = torch.split(normed_w, head_dim, dim=1)
            output = []

            for nx, nc, nw in zip(x_list, c_list, w_list):
                cos_val, sin_val = self.get_cos_sin(nx, nc)
                y0 = torch.mm((nx -  cos_val * self.alpha * nc) * self.scale, nw.t())
                output.append(y0)
            y_TDE = sum(output)
            
        return normed_x*self.scale, normed_w,y_TDE, y_TE


    def get_cos_sin(self, x, y):
        cos_val = (x * y).sum(-1, keepdim=True) / torch.norm(x, 2, 1, keepdim=True) / torch.norm(y, 2, 1, keepdim=True)
        sin_val = (1 - cos_val * cos_val).sqrt()
        return cos_val, sin_val

    def multi_head_call(self, func, x, weight=None):
        #assert len(x.shape) == 2
        x_list = torch.split(x, self.head_dim, dim=1)
        if weight:
            y_list = [func(item, weight) for item in x_list]
        else:
            y_list = [func(item) for item in x_list]
        assert len(x_list) == self.num_head
        assert len(y_list) == self.num_head
        return torch.cat(y_list, dim=1)

    def l2_norm(self, x):
        normed_x = x / torch.norm(x, 2, 1, keepdim=True)
        return normed_x

    def capsule_norm(self, x):
        norm= torch.norm(x.clone(), 2, 1, keepdim=True)
        normed_x = (norm / (1 + norm)) * (x / norm)
        return normed_x

    def causal_norm(self, x, weight):
        norm= torch.norm(x, 2, 1, keepdim=True)
        normed_x = x / (norm + weight)
        return normed_x    

    
    
class Re_Train_DNN_CIFAR10(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_classes):
        super(Re_Train_DNN_CIFAR10, self).__init__()
        self.l1 = nn.Linear(input_dim, hidden_dims[0])
        self.l2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.l3 = nn.Linear(hidden_dims[1], n_classes)

    def forward(self, x):
        h1 = self.l1(x)
        h1 = F.relu(h1)
        h2 = self.l2(h1)

        out = self.l3(h2)

        return h2, out

    
class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Autoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU())
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, output_size))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    
class Re_Train_DNN_Prompt(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_classes, level='c1', yeta=0.1):
        super(Re_Train_DNN_Prompt, self).__init__()
        self.l1 = nn.Linear(input_dim, hidden_dims[0])
        self.l2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.l3 = nn.Linear(hidden_dims[1], n_classes)
        self.yeta = yeta
        self.level = level
        
        if self.level == 'c1':
            self.prompt_embeddings_h = nn.Parameter(torch.zeros(1, 84))
        if self.level == 'c2':
            self.prompt_embeddings_h = nn.Parameter(torch.zeros(1, 84))
            self.prompt_embeddings_x1 = nn.Parameter(torch.zeros(1, 84))
        if self.level == 'c3':
            self.prompt_embeddings_h = nn.Parameter(torch.zeros(1, 84))
            self.prompt_embeddings_x1 = nn.Parameter(torch.zeros(1, 84))     
            self.prompt_embeddings_x2 = nn.Parameter(torch.zeros(1, 512))  
            
    def forward(self, h):
        
        if self.level == 'c1':
            prompt_embeddings_h = self.prompt_embeddings_h.repeat(h.shape[0], 1)
            x1 = self.l1((1 - self.yeta) * h + self.yeta * prompt_embeddings_h)
            x1 = F.relu(x1)
            x2 = self.l2(x1)
            y = self.l3(x2)
        if self.level == 'c2':
            prompt_embeddings_h = self.prompt_embeddings_h.repeat(h.shape[0], 1)
            prompt_embeddings_x1 = self.prompt_embeddings_x1.repeat(h.shape[0], 1)
            x1 = self.l1((1 - self.yeta) * h + self.yeta * prompt_embeddings_h)
            x1 = F.relu(x1)
            x2 = self.l2((1 - self.yeta) * x1 + self.yeta * prompt_embeddings_x1)
            y = self.l3(x2)
        if self.level == 'c3':
            prompt_embeddings_h = self.prompt_embeddings_h.repeat(h.shape[0], 1)
            prompt_embeddings_x1 = self.prompt_embeddings_x1.repeat(h.shape[0], 1) 
            prompt_embeddings_x2 = self.prompt_embeddings_x2.repeat(h.shape[0], 1) 
            x1 = self.l1((1 - self.yeta) * h + self.yeta * prompt_embeddings_h)
            x1 = F.relu(x1)
            x2 = self.l2((1 - self.yeta) * x1 + self.yeta * prompt_embeddings_x1)
            y = self.l3((1 - self.yeta) * x2 + self.yeta * prompt_embeddings_x2)

        return x2, y    
    
    
class Re_Train_DNN_Prompt_v2(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_classes, level='c1', yeta=0.1):
        super(Re_Train_DNN_Prompt_v2, self).__init__()
        self.l1 = nn.Linear(input_dim, hidden_dims[0])
        self.l2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.l3 = nn.Linear(hidden_dims[1], n_classes)
        self.yeta = yeta
        self.level = level
        
        if self.level == 'c1':
            self.prompt_embeddings_h = nn.Parameter(torch.zeros(1, 84))
        if self.level == 'c2':
            self.prompt_embeddings_h = nn.Parameter(torch.zeros(1, 84))
            self.prompt_embeddings_x1 = nn.Parameter(torch.zeros(1, 84))
        if self.level == 'c3':
            self.prompt_embeddings_h = nn.Parameter(torch.zeros(1, 84))
            self.prompt_embeddings_x1 = nn.Parameter(torch.zeros(1, 84))     
            self.prompt_embeddings_x2 = nn.Parameter(torch.zeros(1, 512))  
            
    def forward(self, h):
        
        if self.level == 'c1':
            prompt_embeddings_h = self.prompt_embeddings_h.repeat(h.shape[0], 1)
            x1 = self.l1(h)
            x1 = F.relu(x1)
            x2 = self.l2(x1)
            y = self.l3(x2)
            
        if self.level == 'c2':
            prompt_embeddings_h = self.prompt_embeddings_h.repeat(h.shape[0], 1)
            prompt_embeddings_x1 = self.prompt_embeddings_x1.repeat(h.shape[0], 1)
            x1 = self.l1((1 - self.yeta) * h + self.yeta * prompt_embeddings_h)
            x1 = F.relu(x1)
            x2 = self.l2((1 - self.yeta) * x1 + self.yeta * prompt_embeddings_x1)
            y = self.l3(x2)
        if self.level == 'c3':
            prompt_embeddings_h = self.prompt_embeddings_h.repeat(h.shape[0], 1)
            prompt_embeddings_x1 = self.prompt_embeddings_x1.repeat(h.shape[0], 1) 
            prompt_embeddings_x2 = self.prompt_embeddings_x2.repeat(h.shape[0], 1) 
            x1 = self.l1((1 - self.yeta) * h + self.yeta * prompt_embeddings_h)
            x1 = F.relu(x1)
            x2 = self.l2((1 - self.yeta) * x1 + self.yeta * prompt_embeddings_x1)
            y = self.l3((1 - self.yeta) * x2 + self.yeta * prompt_embeddings_x2)

        return x2, y    
        
    
    