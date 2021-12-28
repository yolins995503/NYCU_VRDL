import os, torch, torchvision, random
import pandas as pd
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image

import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import models
from torch import optim
from torchsummary import summary
import copy
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
lr = 1e-2
epochs = 100
print(device)

class FishDataset(Dataset):
    
    def __init__(self, filenames, labels, transform):
        
        self.filenames = filenames # 資料集的所有檔名
        self.labels = labels # 影像的標籤
        self.transform = transform # 影像的轉換方式
 
    def __len__(self):
        
        return len(self.filenames) # return DataSet 長度
 
    def __getitem__(self, idx):
        
        image = Image.open(self.filenames[idx]).convert('RGB')
        image = self.transform(image) # Transform image
        label = np.array(self.labels[idx])
                
        return image, label # return 模型訓練所需的資訊


def split_train_val_data(data_dir):

    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

    # Transformer
    train_transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    
    test_transformer = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    if args.model == 'inception_v3':
        print('inception_v3')
        train_transformer = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        test_transformer = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            normalize
        ])

    dataset = ImageFolder(data_dir) 
    # 建立 8 類的 list
    character = [[] for i in range(len(dataset.classes))]
    # print(character)
    # 將每一類的檔名依序存入相對應的 list
    for x, y in dataset.samples:
        character[y].append(x)
    
    train_inputs, test_inputs = [], []
    train_labels, test_labels = [], []
    
    for i, data in enumerate(character): # 讀取每個類別中所有的檔名 (i: label, data: filename)
        
        np.random.seed(42)
        np.random.shuffle(data)
            
        # -------------------------------------------
        # 將每一類都以 8:2 的比例分成訓練資料和測試資料
        # -------------------------------------------
        
        num_sample_train = int(len(data)*0.8)
        num_sample_test = len(data) - num_sample_train
        
        # print(str(i) + ': ' + str(len(data)) + ' | ' + str(num_sample_train) + ' | ' + str(num_sample_test))
        
        for x in data[:num_sample_train] : # 前 80% 資料存進 training list
            train_inputs.append(x)
            train_labels.append(i)
            
        for x in data[num_sample_train:] : # 後 20% 資料存進 testing list
            test_inputs.append(x)
            test_labels.append(i)

    train_dataloader = DataLoader(FishDataset(train_inputs, train_labels, train_transformer),
                                  batch_size = batch_size, shuffle = True)
    test_dataloader = DataLoader(FishDataset(test_inputs, test_labels, test_transformer),
                                  batch_size = batch_size, shuffle = False)
 
    return train_dataloader, test_dataloader


def parse_config():
    """Define parse config
    model: which model you would select
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnext101", type=str)

    return parser.parse_args()

def select_model(model_name: str):
    """Select model to predict images
    Parameters:
    -----------
    model_name: str
        Include resnext101, efficientnet b2~b4, resnet50, and regnet.
    Returns:
    -----------
    model:
        Use to train images.
    """
    if model_name == 'resnext101':
        SAVE_MODEL_PATH = 'model/resnext101_batch4_epoch100_best.pth'
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        model = torch.hub.load(
            'pytorch/vision:v0.10.0',
            'resnext101_32x8d',
            pretrained=True
        )
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 7)
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        summary(model, (3, 224, 224))

    elif model_name == 'efficientnet_b4':
        SAVE_MODEL_PATH = 'model/efficient_b4_batch4_epoch100.pth'
        model = models.efficientnet_b4(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 7)
        )
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        summary(model, (3, 224, 224))

    elif model_name == 'resnet50':
        SAVE_MODEL_PATH = 'model/resnet50_batch4_epoch'
        model = models.resnet50(pretrained=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 7)
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        summary(model, (3, 224, 224))

    elif model_name == 'efficientnet_b3':
        SAVE_MODEL_PATH = 'model/efficient_b3_batch4_epoch100.pth'
        model = models.efficientnet_b3(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 7)
        )
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        summary(model, (3, 224, 224))

    elif model_name == 'efficientnet_b2':
        SAVE_MODEL_PATH = 'model/efficient_b2_batch4_epoch100.pth'
        model = models.efficientnet_b2(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 7)
        )
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        summary(model, (3, 224, 224))

    elif model_name == 'regnet_x_8gf':
        SAVE_MODEL_PATH = 'model/regnet_x_8gf_batch4_epoch100_best.pth'
        model = models.regnet_x_8gf(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 7)
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        summary(model, (3, 224, 224))

    elif model_name == 'inception_v3':
        SAVE_MODEL_PATH = 'model/inception_v3_batch4_epoch100_best.pth'
        model = models.inception_v3(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 7)
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        summary(model, (3, 299, 299))
    
    elif model_name == 'regnet_x_16gf':
        SAVE_MODEL_PATH = 'model/regnet_x_16gf_batch4_epoch100_best.pth'
        model = models.regnet_x_16gf(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 7)
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        summary(model, (3, 224, 224))
    
    elif model_name == 'regnet_x_32gf':
        SAVE_MODEL_PATH = 'model/regnet_x_32gf_batch4_epoch100_best.pth'
        model = models.regnet_x_32gf(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 7)
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        summary(model, (3, 224, 224))

    
    return model, optimizer, SAVE_MODEL_PATH


if __name__ == '__main__':

    args = parse_config()
    model, optimizer, SAVE_MODEL_PATH = select_model(args.model)
    criterion = nn.CrossEntropyLoss()

    loss_epoch = []
    train_acc, test_acc = [], []
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    train_dataloader, test_dataloader = split_train_val_data('train_crop')

    # Modify by yourself
    step_lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=25,
        gamma=0.1
    )

    for epoch in range(epochs):
        iter = 0
        correct_train, total_train = 0, 0
        correct_test, total_test = 0, 0
        train_loss = 0.0

        model.train()
        print('epoch: ' + str(epoch + 1) + ' / ' + str(epochs))

        # ---------------------------
        # Training Stage
        # ---------------------------

        for i, (x, label) in enumerate(train_dataloader):
            x = x.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.int64)

            optimizer.zero_grad()
            if args.model == 'inception_v3':
                outputs, aux_outputs = model(x)
                loss1 = criterion(outputs, label)
                loss2 = criterion(aux_outputs, label)
                loss = loss1 + 0.4*loss2
            else:
                outputs = model(x)
                loss = criterion(outputs, label)
            
            loss.backward()
            optimizer.step()

            # calculate training data accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += label.size(0)
            correct_train += (predicted == label).sum()

            train_loss += loss.item()
            iter += 1

        print(
            'Training epoch: %d / loss: %.3f | acc: %.3f' %
            (epoch + 1, train_loss / iter, correct_train / total_train)
        )

        # --------------------------
        # Testing Stage
        # --------------------------

        model.eval()

        for i, (x, label) in enumerate(test_dataloader):
            with torch.no_grad():  # don't need gradient
                x, label = x.to(device), label.to(device)
                outputs = model(x)  # predict image
                # calculate testing data accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_test += label.size(0)
                correct_test += (predicted == label).sum()

        valid_acc = correct_test / total_test
        print(f'Validation acc: {valid_acc:.3f}')
        if (epoch + 1) % 10 == 0:
            best_model_wts = copy.deepcopy(model.state_dict())
            best_acc = valid_acc
            torch.save(best_model_wts, f'{SAVE_MODEL_PATH}{epoch+1}_best.pth')

        step_lr_scheduler.step()

        train_acc.append(100 * (correct_train / total_train))
        test_acc.append(100 * (correct_test / total_test))
        loss_epoch.append(train_loss / iter)

    plt.figure()
    x = np.linspace(1, epochs, epochs)
    loss_epoch = np.array(loss_epoch)
    plt.plot(x, loss_epoch)  # plot your loss

    plt.title('Training Loss')
    plt.ylabel('loss'), plt.xlabel('epoch')
    plt.legend(['loss'], loc='upper left')
    plt.show()

    plt.figure()

    train_acc = np.array(train_acc)
    test_acc = np.array(test_acc)

    plt.plot(x, train_acc)  # plot your training accuracy
    plt.plot(x, test_acc)  # plot your testing accuracy

    plt.title('Training acc')
    plt.ylabel('acc (%)'), plt.xlabel('epoch')
    plt.legend(['training acc', 'valid acc'], loc='upper left')
    plt.show()