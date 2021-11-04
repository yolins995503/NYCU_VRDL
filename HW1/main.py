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
from torchsummary import summary
import pretrainedmodels.models.resnext as resnext
from tqdm import tqdm

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
        data_augmentation.append(transforms.RandomHorizontalFlip(p=0.5))
        #data_augmentation.append(transforms.RandomVerticalFlip(p=0.5))
        data_augmentation.append(transforms.RandomRotation(10))
        #data_augmentation.append(transforms.RandomCrop(196,padding=2))
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
                content = line.rstrip().split(' ')
                name = content[0]
                labels = []
                for value in content[1:]:
                    head, sep, tail = value.partition('.')
                    labels.append(int(head)-1)
                image_label_list.append((name,labels))
        return image_label_list

    def load_data(self , path , resize_height , resize_width , normalization):
        image = image_processing.read_image(path , resize_height , resize_width ,normalization)
        return image
    
    def data_preproccess(self , data):
        #data = self.toTensor(data)
        #data = torch.tensor(data)
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

class ResNet101(nn.Module):
   def __init__(self,num_class,pretrained_option=False):
        super(ResNet101,self).__init__()
        self.model=models.resnet101(pretrained=pretrained_option)
        
        if pretrained_option==True:
            for param in self.model.parameters():
                param.requires_grad=False
        num_neurons=self.model.fc.in_features
        self.model.fc=nn.Linear(num_neurons,num_class)
        
   def forward(self,X):
        out=self.model(X)
        return out

def training(model, train_loader, test_loader, Loss, optimizer, epochs, device, num_class, name):
    model.to(device)
    best_model_wts = None
    best_evaluated_acc = 0
    train_acc = []
    test_acc = []
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer , gamma = 0.96)
    for epoch in range(1, epochs+1):
        with torch.set_grad_enabled(True):
            model.train()
            total_loss=0
            correct=0
            for idx,(data, label) in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()
                        
                data = data.to(device,dtype=torch.float)
                label = label.to(device,dtype=torch.long)

                predict = model(data)      

                loss = Loss(predict, label.squeeze())

                total_loss += loss.item()
                pred = torch.max(predict,1).indices
                correct += pred.eq(label).cpu().sum().item()
                        
                loss.backward()
                optimizer.step()

            total_loss /= len(train_loader.dataset)
            correct = (correct/len(train_loader.dataset))*100.
            train_acc.append(correct)
            print ("Epoch : " , epoch)
            print ("Loss : " , total_loss)
            print ("Correct : " , correct)
            #print(epoch, total_loss, correct)     
        scheduler.step()
        accuracy = evaluate(model, device, test_loader)  
        test_acc.append(accuracy)
        print ("Accuracy : " , accuracy ,"%")
        print ("---------------------------------------------------------")

        if accuracy > best_evaluated_acc:
            best_evaluated_acc = accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
    #save model
    torch.save(best_model_wts, name+".pt")
    model.load_state_dict(best_model_wts)
    return train_acc, test_acc 

def evaluate(model, device, test_loader):
    correct=0
    with torch.set_grad_enabled(False):
        model.eval()
        for idx,(data,label) in enumerate(test_loader):
            data = data.to(device,dtype=torch.float)
            label = label.to(device,dtype=torch.long)
            predict = model(data)
            pred = torch.max(predict,1).indices
            #correct += pred.eq(label).cpu().sum().item()
            for j in range(data.size()[0]):
                #print ("{} pred label: {} ,true label:{}" .format(len(pred),pred[j],int(label[j])))
                if (int (pred[j]) == int (label[j])):
                    correct +=1
        print ("num_correct :",correct ," / " , len(test_loader.dataset))
        correct = (correct/len(test_loader.dataset))*100.
    return correct 

def show_curve(list_array,name):
    plt.title(name)
    epoch = []
    for i in range(len(list_array)):
        epoch.append(i)
    plt.plot(epoch,list_array)
    plt.savefig(name+'.png')
    #plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-model_train" , type = int , default = 1)
    parser.add_argument("-model_test" , type = int , default = 0)

    parser.add_argument("-model_pretrain" , type = int , default = 1)
    parser.add_argument("-pretrain_model_weight" , type = str , default = "resnet101-63fe2227.pth") 

    parser.add_argument("-model_name" , type = str , default = 'CNN_model')
    parser.add_argument("-train_filename" , type = str , default='./2021VRDL_HW1_datasets/training_labels.txt')
    parser.add_argument("-valid_filename" , type = str , default='./2021VRDL_HW1_datasets/training_labels.txt')
    parser.add_argument("-test_filename" , type = str , default='./2021VRDL_HW1_datasets/training_labels.txt')

    parser.add_argument("-image_dir_train" , type = str , default='./2021VRDL_HW1_datasets/training_images')
    parser.add_argument("-image_dir_valid" , type = str , default='./2021VRDL_HW1_datasets/training_images')
    parser.add_argument("-image_dir_test" , type = str , default='./2021VRDL_HW1_datasets/testing_images')

    parser.add_argument("-num_class" , type = int , default = 200)
    parser.add_argument("-epoch" , type = int , default = 30)
    parser.add_argument("-batch_size" , type = int , default = 16)
    parser.add_argument("-Learning_rate" , type = int , default = 1e-3)
    parser.add_argument("-momentum" , type = int , default = 0.9)
    parser.add_argument("-weight_decay" , type = int , default = 5e-4)
    
    args = parser.parse_args()

    train_data = TorchDataset(filename=args.train_filename, image_dir=args.image_dir_train,repeat=1)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False)

    valid_data = TorchDataset(filename=args.valid_filename, image_dir=args.image_dir_valid,repeat=1)
    valid_loader = DataLoader(dataset=valid_data, batch_size=args.batch_size,shuffle=False)

    test_data = TorchDataset(filename=args.test_filename, image_dir=args.image_dir_test,repeat=1)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size,shuffle=False)

    print (train_loader)
    dataiter = iter(train_loader)
    images , labels = dataiter.next()
    print (type(images) , type(labels))
    print (images.size(),labels.size())

    #check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device:
        print ("GPU is ready")

    if args.model_pretrain == 0:
        model = ResNet50(args.num_class, False)

    if args.model_pretrain == 1:
        '''
        model = models.resnet101(pretrained=False)
        for param in model.parameters():
            param.requires_grad=True
        model.load_state_dict(torch.load(args.pretrain_model_weight))
        num_neurons=model.fc.in_features
        #model.fc=nn.Linear(num_neurons,num_neurons)
        model.fc=nn.Linear(num_neurons,args.num_class)
        '''

        '''
        model = models.resnext101_32x8d(pretrained=False)
        for param in model.parameters():
            param.requires_grad=True
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(args.pretrain_model_weight))
        num_neurons=model.fc.in_features
        #model.fc=nn.Linear(num_neurons,num_neurons)
        model.fc=nn.Linear(num_neurons,args.num_class)
        '''

        model = resnext.resnext101_64x4d(1000,pretrained='imagenet')
        for param in model.parameters():
            param.requires_grad=True
        #model.load_state_dict(torch.load(args.pretrain_model_weight))
        model.last_linear = nn.Linear(2048,args.num_class) 
        

    if args.model_train == 1:
        #train
        print ("Train Start")
        #summary(model.to(device), (3, 224, 224))
        Loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.Learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        train_acc , test_acc  = training(model, train_loader, valid_loader, Loss, optimizer,args.epoch, device, args.num_class, args.model_name)

        with open('model_acc.txt', 'w') as file:
            for i in range(len(train_acc)):
                file.write(str(test_acc[i]))
                file.write("\n")
            file.close()

        show_curve(test_acc , "Accuracy")

    if args.model_test == 1:
        #test
        print ("Test Start")
        if args.model_pretrain == 0:
            model = ResNet50(args.num_class, False)
        if args.model_pretrain == 1:
            model = models.resnet50(pretrained=False)
            num_neurons=model.fc.in_features
            model.fc=nn.Linear(num_neurons,args.num_class)
        best_model = model
        best_model.load_state_dict(torch.load("CNN_model.pt"))
        best_model.to(device)
        best_model.eval()
        accuracy = evaluate(best_model, device, valid_loader)
        print ("Accuracy : " , accuracy ,"%")



