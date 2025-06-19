import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.models as models
from resnetcifar import *
from transformer_layers import *
import numpy as np
import copy

class SimpleCNN_header(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN_header, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x




class ModelFedCon(nn.Module):

    def __init__(self, base_model, out_dim, n_classes):
        super(ModelFedCon, self).__init__()

        if base_model == "resnet50-cifar10" or base_model == "resnet50-cifar100" or base_model == "resnet50-smallkernel" or base_model == "resnet50":
            basemodel = ResNet50_cifar10()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == "resnet18-cifar10" or base_model == "resnet18":
            basemodel = ResNet18_cifar10()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
            num_ftrs = 512
        elif base_model == 'simple-cnn':
            self.features = SimpleCNN_header(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=n_classes)
            num_ftrs = 84

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

        # last layer
        self.l3 = nn.Linear(out_dim, n_classes)

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            #print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()
        x1 = self.l1(h)
        x1 = F.relu(x1)
        x2 = self.l2(x1)

        y = self.l3(x2)
        return h, x2, y

class ModelFedCon_text(nn.Module):

    def __init__(self, base_model, out_dim, prototypes):
        super(ModelFedCon_text, self).__init__()

        if base_model == "resnet50-cifar10" or base_model == "resnet50-cifar100" or base_model == "resnet50-smallkernel" or base_model == "resnet50":
            basemodel = ResNet50_cifar10()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == "resnet18-cifar10" or base_model == "resnet18":
            basemodel = ResNet18_cifar10()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
            num_ftrs = 512
        elif base_model == 'simple-cnn':
            self.features = SimpleCNN_header(input_dim=(16 * 5 * 5), hidden_dims=[120, 84])
            num_ftrs = 84

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.prototypes = prototypes
        
    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            #print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()
        x1 = self.l1(h)
        x1 = F.relu(x1)
        x2 = self.l2(x1)

        image_features = x2 / x2.norm(dim=-1, keepdim=True)
        text_features = self.prototypes / self.prototypes.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()      
        
        return logits_per_image
    
    
class ModelFedCon_att(nn.Module):

    def __init__(self, base_model, out_dim, n_classes):
        super(ModelFedCon_att, self).__init__()

        if base_model == "resnet50-cifar10" or base_model == "resnet50-cifar100" or base_model == "resnet50-smallkernel" or base_model == "resnet50":
            basemodel = ResNet50_cifar10()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == "resnet18-cifar10" or base_model == "resnet18":
            basemodel = ResNet18_cifar10()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
            num_ftrs = 512
        elif base_model == 'simple-cnn':
            self.features = SimpleCNN_header(input_dim=(16 * 5 * 5), hidden_dims=[120, 512], output_dim=n_classes)
            num_ftrs = 512

        # projection MLP
        
        self.att = SelfAttnLayer(512, 4, [256, 512], 10)

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            #print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()

        x2, y = self.att(h)
        return h, x2, y
    
    
    
class ModelFedCon_prompt(nn.Module):

    def __init__(self, base_model, out_dim, n_classes, level='c1', yeta=0.1):
        super(ModelFedCon_prompt, self).__init__()
        
        if base_model == "resnet50-cifar10" or base_model == "resnet50-cifar100" or base_model == "resnet50-smallkernel" or base_model == "resnet50":
            basemodel = ResNet50_cifar10()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == "resnet18-cifar10" or base_model == "resnet18":
            basemodel = ResNet18_cifar10()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
            num_ftrs = 512
        elif base_model == 'simple-cnn':
            self.features = SimpleCNN_header(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=n_classes)
            num_ftrs = 84
        
        self.level = level
        
        if self.level == 'c1':
            self.prompt_embeddings_h = nn.Parameter(torch.zeros(1, 84))
            nn.init.normal_(self.prompt_embeddings_h.data, mean=0.0, std=0.5)    
        if self.level == 'c2':
            self.prompt_embeddings_h = nn.Parameter(torch.zeros(1, 84))
            nn.init.normal_(self.prompt_embeddings_h.data, mean=0.0, std=0.5)    
            self.prompt_embeddings_x1 = nn.Parameter(torch.zeros(1, 84))
            nn.init.normal_(self.prompt_embeddings_x1.data, mean=0.0, std=0.5)    
        if self.level == 'c3':
            self.prompt_embeddings_h = nn.Parameter(torch.zeros(1, 84))
            nn.init.normal_(self.prompt_embeddings_h.data, mean=0.0, std=0.5)    
            self.prompt_embeddings_x1 = nn.Parameter(torch.zeros(1, 84))   
            nn.init.normal_(self.prompt_embeddings_x1.data, mean=0.0, std=0.5) 
            self.prompt_embeddings_x2 = nn.Parameter(torch.zeros(1, 512))   
            nn.init.normal_(self.prompt_embeddings_x2.data, mean=0.0, std=0.5)    
                        
        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

        # last layer
        self.l3 = nn.Linear(out_dim, n_classes)        
        self.yeta = yeta
        
    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            #print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        
        h = self.features(x)
        h = h.squeeze()
        if self.level == 'c1':
            prompt_embeddings_h = self.prompt_embeddings_h.repeat(x.shape[0], 1)
            x1 = self.l1((1 - self.yeta) * h + self.yeta * prompt_embeddings_h)
            x1 = F.relu(x1)
            x2 = self.l2(x1)
            y = self.l3(x2)
        if self.level == 'c2':
            prompt_embeddings_h = self.prompt_embeddings_h.repeat(x.shape[0], 1)
            prompt_embeddings_x1 = self.prompt_embeddings_x1.repeat(x.shape[0], 1)
            x1 = self.l1((1 - self.yeta) * h + self.yeta * prompt_embeddings_h)
            x1 = F.relu(x1)
            x2 = self.l2((1 - self.yeta) * x1 + self.yeta * prompt_embeddings_x1)
            y = self.l3(x2)
        if self.level == 'c3':
            prompt_embeddings_h = self.prompt_embeddings_h.repeat(x.shape[0], 1)
            prompt_embeddings_x1 = self.prompt_embeddings_x1.repeat(x.shape[0], 1) 
            prompt_embeddings_x2 = self.prompt_embeddings_x2.repeat(x.shape[0], 1) 
            x1 = self.l1((1 - self.yeta) * h + self.yeta * prompt_embeddings_h)
            x1 = F.relu(x1)
            x2 = self.l2((1 - self.yeta) * x1 + self.yeta * prompt_embeddings_x1)
            y = self.l3((1 - self.yeta) * x2 + self.yeta * prompt_embeddings_x2)
        return h, x2, y
        

class ModelFedCon_prompt_v2(nn.Module):

    def __init__(self, base_model, out_dim, n_classes):
        super(ModelFedCon_prompt_v2, self).__init__()

        if base_model == "resnet50-cifar10" or base_model == "resnet50-cifar100" or base_model == "resnet50-smallkernel" or base_model == "resnet50":
            basemodel = ResNet50_cifar10()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == "resnet18-cifar10" or base_model == "resnet18":
            basemodel = ResNet18_cifar10()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
            num_ftrs = 512
        elif base_model == 'simple-cnn':
            self.features = SimpleCNN_header(input_dim=(16 * 5 * 5), hidden_dims=[120, 512], output_dim=n_classes)
            num_ftrs = 512

        self.prompt_embeddings = nn.Parameter(torch.zeros(1, 512))
        nn.init.uniform_(self.prompt_embeddings.data, -0.1, 0.1)            
                        
        # projection MLP
        self.l1 = nn.Linear(num_ftrs*2, num_ftrs // 2)
        self.l2 = nn.Linear(num_ftrs // 2, out_dim)

        # last layer
        self.l3 = nn.Linear(out_dim, n_classes)        

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            #print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        prompt_embeddings = self.prompt_embeddings.repeat(x.shape[0], 1)
        h = self.features(x)
        h = h.squeeze()
        h = torch.cat([h, prompt_embeddings], dim=1)
        
        x1 = self.l1(h)
        x1 = F.relu(x1)
        x2 = self.l2(x1)

        y = self.l3(x2)
        return h, x2, y        
        
        
    
    
    
class ModelFedCon_noheader(nn.Module):

    def __init__(self, base_model, out_dim, n_classes, net_configs=None):
        super(ModelFedCon_noheader, self).__init__()

        if base_model == "resnet50":
            basemodel = models.resnet50(pretrained=False)
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
#         elif base_model == "resnet18":
#             basemodel = models.resnet18(pretrained=False)
#             self.features = nn.Sequential(*list(basemodel.children())[:-1])
#             num_ftrs = basemodel.fc.in_features
        elif base_model == "resnet50-cifar10" or base_model == "resnet50-cifar100" or base_model == "resnet50-smallkernel":
            basemodel = ResNet50_cifar10()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == "resnet18":
            basemodel = ResNet18_cifar10()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == "mlp":
            self.features = MLP_header()
            num_ftrs = 512
        elif base_model == 'simple-cnn':
            self.features = SimpleCNN_header(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=n_classes)
            num_ftrs = 84
        elif base_model == 'simple-cnn-mnist':
            self.features = SimpleCNNMNIST_header(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=n_classes)
            num_ftrs = 84

        #summary(self.features.to('cuda:0'), (3,32,32))
        #print("features:", self.features)
        # projection MLP
        # self.l1 = nn.Linear(num_ftrs, num_ftrs)
        # self.l2 = nn.Linear(num_ftrs, out_dim)

        # last layer
        self.l3 = nn.Linear(num_ftrs, n_classes)

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            #print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        #print("h before:", h)
#         print("h size:", h.size())
        h = h.squeeze()
        #print("h after:", h)
        # x = self.l1(h)
        # x = F.relu(x)
        # x = self.l2(x)

        y = self.l3(h)
        return h, h, y

class ModelFedCon_ETF(nn.Module):

    def __init__(self, base_model, out_dim, n_classes, method):
        super(ModelFedCon_ETF, self).__init__()

        if base_model == "resnet50-cifar10" or base_model == "resnet50-cifar100" or base_model == "resnet50-smallkernel" or base_model == "resnet50":
            basemodel = ResNet50_cifar10()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == "resnet18-cifar10" or base_model == "resnet18":
            basemodel = ResNet18_cifar10()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == "mlp":
            self.features = MLP_header()
            num_ftrs = 512
        elif base_model == 'simple-cnn':
            self.features = SimpleCNN_header(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=n_classes)
            num_ftrs = 84
        elif base_model == 'simple-cnn-mnist':
            self.features = SimpleCNNMNIST_header(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=n_classes)
            num_ftrs = 84

        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

        # last layer
        self.proto_classifier = Proto_Classifier(out_dim, n_classes)
        self.scaling_train = torch.nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()     
        h = F.relu(self.l1(h))
        f = self.l2(h)
        f_norm = torch.norm(f, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        f = torch.div(f, f_norm)        

        return h, f, f
    
    
class Proto_Classifier(nn.Module):
    def __init__(self, feat_in, num_classes):
        super(Proto_Classifier, self).__init__()
        P = self.generate_random_orthogonal_matrix(feat_in, num_classes)
        I = torch.eye(num_classes)
        one = torch.ones(num_classes, num_classes)
        M = np.sqrt(num_classes / (num_classes-1)) * torch.matmul(P, I-((1/num_classes) * one))

        self.proto = M.cuda()

    def generate_random_orthogonal_matrix(self, feat_in, num_classes):
        a = np.random.random(size=(feat_in, num_classes))
        P, _ = np.linalg.qr(a)
        P = torch.tensor(P).float()
        assert torch.allclose(torch.matmul(P.T, P), torch.eye(num_classes), atol=1e-06), torch.max(torch.abs(torch.matmul(P.T, P) - torch.eye(num_classes)))
        return P

    def load_proto(self, proto):
        self.proto = copy.deepcopy(proto)

    def forward(self, label):
        # produce the prototypes w.r.t. the labels
        target = self.proto[:, label].T ## B, d  output: B, d
        return target

    
    