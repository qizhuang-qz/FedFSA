import torch.utils.data as data
from PIL import Image
import numpy as np
import torchvision
from torchvision.datasets import MNIST, EMNIST, CIFAR10, CIFAR100, SVHN, FashionMNIST, ImageFolder, DatasetFolder, utils
from torch.utils.data import Dataset
import os
import os.path
import logging
import sys
import torch
import io
import scipy.io as matio
import ipdb
from torchvision import transforms
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass
name_classes = np.array(["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])
    
class CustomDataset(Dataset):
    def __init__(self, img_dir, missing_list, transform_local=None, target_transform=None):
        self.img_dir = img_dir
        self.missing_list = missing_list
        self.name_classes = name_classes
        self.transform_local = transform_local
        self.target_transform = target_transform
        self.data, self.target = self.__build_Custom_dataset__()

    def __build_Custom_dataset__(self):
        # Initialize data lists
        all_images = []
        all_labels = []

        # Extract images and labels from directory
        label_names = self.name_classes[np.array(self.missing_list, dtype=np.int)]

        for label_name in label_names:
            if label_name != '.ipynb_checkpoints':
                label = np.where(self.name_classes == label_name)[0][0]
#                 ipdb.set_trace()
                class_folder = os.path.join(self.img_dir, label_name)
                k = 0
                for img_name in os.listdir(class_folder):
                    if img_name.endswith('.jpg'):
                        img_path = os.path.join(class_folder, img_name)
                        image = Image.open(img_path).convert('RGB')
                        image = transform(image)
                        all_images.append(image)
                        all_labels.append(label)
#                         if k >= 200:
#                             break
#                         else:
#                             k += 1

        # Convert lists to tensors
        all_images_tensor = np.stack(all_images)
        all_labels_tensor = np.array(all_labels)
        print(all_images_tensor.shape)
        return all_images_tensor, all_labels_tensor

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        
        if self.transform_local is not None:
            img = self.transform_local(img)            
            
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)    
    
    
    
    
    
    
    
def default_loader(image_path):
    return Image.open(image_path).convert('RGB')  


# class CustomDataset(Dataset):
#     def __init__(self, img_dir, transform=None, missing=None):
#         self.img_dir = img_dir
#         self.transform = transform
#         # 递归遍历文件夹下的所有.jpg文件
#         self.data = []
#         self.target = []

#         for class_id in range(10):
#             class_folder = os.path.join(img_dir, f'client_{client_id}', f'class_{class_id}')
#             for img_name in os.listdir(class_folder):
#                 if img_name.endswith('.jpg'):
#                     self.data.append(os.path.join(class_folder, img_name))
#                     self.target.append(class_id)
        
        
#     def __len__(self):
#         return len(self.target)

#     def __getitem__(self, idx):
#         img_path, label = self.data[idx], self.target[idx]
#         image = Image.open(img_path).convert('RGB')
        
#         if self.transform:
#             image = self.transform(image)

#         return image, label

def transform(img):
    img = img.resize((32, 32))  # 调整图像大小
    img = np.array(img)  # 将图像转换为tensor
#     img = img.permute(2, 0, 1)  # 将维度顺序调整为[3, 32, 32]
    return img


class CIFAR10_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform_local=None, transform_clip=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform_local = transform_local
        self.transform_clip = transform_clip
        self.target_transform = target_transform
        self.download = download
        self.data, self.target = self.__build_truncated_dataset__()
        
    def __build_truncated_dataset__(self):

        cifar_dataobj = CIFAR10(self.root, self.train, self.transform_local, self.target_transform, self.download)

        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = cifar_dataobj.train_data, np.array(cifar_dataobj.train_labels)
            else:
                data, target = cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
            
            
#         for i in range(data.shape[0]):
#             image = Image.fromarray(data[i])
#             image.save('t-SNE images/cifar10/'+str(i)+'.jpg')
#             if i > 100:
#                 ipdb.set_trace()
        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        # img = Image.fromarray(img)
        # print("cifar10 img:", img)
        # print("cifar10 target:", target)

#         if self.transform_clip is not None:
#             img_clip = self.transform_clip(img)
        if self.transform_local is not None:
            img_local = self.transform_local(img)            
            
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img_local, target

    def __len__(self):
        return len(self.data)


class CIFAR100_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform_local=None, transform_clip=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform_local = transform_local
        self.transform_clip = transform_clip
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        cifar_dataobj = CIFAR100(self.root, self.train, self.transform_local, self.target_transform, self.download)

        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = cifar_dataobj.train_data, np.array(cifar_dataobj.train_labels)
            else:
                data, target = cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        img = Image.fromarray(img)
        # print("cifar10 img:", img)
        # print("cifar10 target:", target)

        if self.transform_local is not None:
            img_local = self.transform_local(img)
#         if self.transform_clip is not None:
#             img_clip = self.transform_clip(img)            

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img_local, target

    def __len__(self):
        return len(self.data)

class TinyImageNet_load(Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform_local=None, transform_clip=None):
        self.Train = train
        self.root_dir = root
        self.transform_local = transform_local
        self.transform_clip = transform_clip
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")
        self.dataidxs = dataidxs
        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        
        if self.dataidxs is not None:
            self.samples = self.images[dataidxs]
        else:
            self.samples = self.images

#         print('samples.shape', self.samples.shape)
        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images;

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
#         print(self.tgt_idx_to_class)
    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(val_image_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)
        self.images = np.array(self.images)
#         print('dataset.shape', self.images.shape)
    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]
    

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        img_path, tgt = self.samples[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
#         if self.transform_clip is not None:
#             sample_clip = self.transform_clip(sample)
        if self.transform_local is not None:
            sample_local = self.transform_local(sample)            
        tgt = int(tgt)
        return sample_local, tgt     

