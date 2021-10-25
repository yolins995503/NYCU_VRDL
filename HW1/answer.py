import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import image_processing
import os
import csv
import copy
import argparse
import torch
import torch.nn as nn
from torchvision import models
import time
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from torchsummary import summary
import pretrainedmodels.models.resnext as resnext

class TorchDataset(Dataset):
    def __init__(self, filename , image_dir , resize_height=224, resize_width=224 ,repeat = 1, augmentation=None):
        # --------------------------------------------
        # Initialize paths, transforms, and so on
        # --------------------------------------------
        self.image_label_list = self.read_file(filename)
        self.image_dir = image_dir
        self.len = len(self.image_label_list)
        self.repeat = repeat
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.toTensor = transforms.ToTensor()

        data_augmentation = []
        #data_augmentation = [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5),transforms.ToTensor()]
        #data_augmentation.append(transforms.RandomHorizontalFlip(p=0.5))
        #data_augmentation.append(transforms.RandomVerticalFlip(p=0.5))
        #data_augmentation.append(transforms.RandomRotation(5))
        #data_augmentation.append(transforms.RandomCrop(32,padding=2))
        data_augmentation.append(transforms.ToTensor())
        data_augmentation.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))

        self.transformations=transforms.Compose(data_augmentation)

    def __getitem__(self, i):
        # --------------------------------------------
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        # --------------------------------------------
        index = i % self.len
        # print("i={},index={}".format(i, index))
        image_name, label = self.image_label_list[index]
        image_path = os.path.join(self.image_dir, image_name)
        img = self.load_data(image_path, self.resize_height, self.resize_width, normalization=False)
        img = self.data_preproccess(img)
        label=np.array(label)
        return img, label

    def __len__(self):
        # --------------------------------------------
        # Indicate the total size of the dataset
        # --------------------------------------------
        if self.repeat == None:
            data_len = 10000000
        else:
            data_len = len(self.image_label_list) * self.repeat
        return data_len

    def read_file(self , filename):
        image_label_list = []
        with open(filename,'r') as f:
            lines = f.readlines()
            for line in lines:
                
                # content = line.rstrip().split(' ')
                # name = content[0]
                # labels = []
                # for value in content[1:]:
                #     head, sep, tail = value.partition('.')
                #     labels.append(int(head)-1)
                # image_label_list.append((name,labels))
                
                
                content = line.rstrip().split(' ')
                name = content[0]
                labels = []
                labels.append(0)
                image_label_list.append((name,labels))
                

        return image_label_list

    def load_data(self , path , resize_height , resize_width , normalization):
        image = image_processing.read_image(path , resize_height , resize_width ,normalization)
        return image
    
    def data_preproccess(self , data):
        #data = self.toTensor(data)
        data = self.transformations(data)
        return data

class ResNet50(nn.Module):
   def __init__(self,num_class,pretrained_option=False):
        super(ResNet50,self).__init__()
        self.model=models.resnet50(pretrained=pretrained_option)
        
        if pretrained_option==True:
            for param in self.model.parameters():
                param.requires_grad=False
        num_neurons=self.model.fc.in_features
        self.model.fc=nn.Linear(num_neurons,num_class)
        
   def forward(self,X):
        out=self.model(X)
        return out

def evaluate(model, device, test_loader):
    correct=0
    result = []
    with torch.set_grad_enabled(False):
        model.eval()
        for idx,(data,label) in enumerate(test_loader):
            data = data.to(device,dtype=torch.float)
            label = label.to(device,dtype=torch.long)
            predict = model(data)
            pred = torch.max(predict,1).indices
            #correct += pred.eq(label).cpu().sum().item()
            for j in range(data.size()[0]):
                #print ("{} pred, label: {} , true label:{}" .format(len(pred),pred[j],int(label[j])))
                result.append(int(pred[j]))
                if (int (pred[j]) == int (label[j])):
                    correct +=1
        print ("num_correct :",correct)
        correct = (correct/len(test_loader.dataset))*100.
    return correct , result

def write_answer(result):
    with open('result.txt','w') as f:
        for i in range(len(result)):
            f.write(str(result[i]+1))
            f.write("\n")
        f.close()
    name_list = []
    with open ("./2021VRDL_HW1_datasets/classes.txt", "r") as f:
        for line in f.readlines():
            line = line.rstrip("\n")
            name_list.append(line)
        f.close()
    with open ("answer.txt","w") as f1:
        with open("result.txt" , "r") as f2:
            with open ("./2021VRDL_HW1_datasets/testing_img_order.txt" , "r") as f3:
                for line3 , line2 in zip(f3.readlines(),f2.readlines()):
                    image_name = line3.rstrip("\n")
                    image_predict = int(line2.rstrip("\n"))
                    image_predict = name_list[image_predict-1]
                    f1.write(image_name)
                    f1.write(" ")
                    f1.write(image_predict)
                    f1.write("\n")
                f3.close()
            f2.close()
        f1.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-model_name" , type = str , default = 'CNN_model')

    parser.add_argument("-model_pretrain" , type = int , default = 1)
    parser.add_argument("-pretrain_model_weight" , type = str , default = "resnet50-0676ba61.pth") 

    parser.add_argument("-valid_filename" , type = str , default='./2021VRDL_HW1_datasets/training_labels.txt')
    parser.add_argument("-test_filename" , type = str , default='./2021VRDL_HW1_datasets/testing_img_order.txt')

    parser.add_argument("-image_dir_valid" , type = str , default='./2021VRDL_HW1_datasets/training_images')
    parser.add_argument("-image_dir_test" , type = str ,default='./2021VRDL_HW1_datasets/testing_images')

    parser.add_argument("-num_class" , type = int , default = 200)
    parser.add_argument("-epoch" , type = int , default = 100)
    parser.add_argument("-batch_size" , type = int , default = 64)
    
    args = parser.parse_args()

    valid_data = TorchDataset(filename=args.valid_filename, image_dir=args.image_dir_valid,repeat=1)
    valid_loader = DataLoader(dataset=valid_data, batch_size=args.batch_size,shuffle=False)

    test_data = TorchDataset(filename=args.test_filename, image_dir=args.image_dir_test,repeat=1)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size,shuffle=False)

    #check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device:
        print ("GPU is ready")

    if args.model_pretrain == 0:
        model = ResNet50(args.num_class, False)

    #test
    print ("Test Start")
    if args.model_pretrain == 0:
        model = ResNet50(args.num_class, False)
    if args.model_pretrain == 1:
        '''
        model = models.resnet101(pretrained=False)
        num_neurons=model.fc.in_features
        #model.fc=nn.Linear(num_neurons,num_neurons)
        model.fc=nn.Linear(num_neurons,args.num_class)
        '''

        model = resnext.resnext101_64x4d(1000,pretrained='imagenet')
        model.last_linear = nn.Linear(2048,args.num_class) 

    best_model = model
    best_model.load_state_dict(torch.load("CNN_model.pt"))
    best_model.to(device)
    best_model.eval()
    accuracy , result = evaluate(best_model, device, test_loader)
    print ("Accuracy : " , accuracy ,"%")

    write_answer(result)







