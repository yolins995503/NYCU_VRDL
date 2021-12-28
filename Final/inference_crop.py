import os
import torch
import torchvision
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
from torchvision import models
import gc
import pandas as pd
from tqdm import tqdm
from glob import glob

def select_model(model_name: str):
    """Select model to predict images
    Parameters:
    -----------
    model_name: str
        Include resnext101, efficientnet b2~b4, resnet50, and regnet.
    Returns:
    -----------
    model:
        Use to predict images.
    """
    if model_name == 'resnext101':
        LOAD_MODEL_PATH = 'model/resnext101_batch4_epoch100_best.pth'
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        model = torch.hub.load(
            'pytorch/vision:v0.10.0',
            'resnext101_32x8d',
            pretrained=False
        )
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 7)
        model.load_state_dict(torch.load(LOAD_MODEL_PATH))
        model = model.to(device)

    elif model_name == 'resnet50':
        LOAD_MODEL_PATH = 'model/resnet50_batch4_epoch10_best.pth'
        model = models.resnet50(pretrained=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 7)
        model.load_state_dict(torch.load(LOAD_MODEL_PATH))
        model = model.to(device)

    elif model_name == 'efficientnet_b2':
        LOAD_MODEL_PATH = 'model/efficient_b2_batch4_epoch100.pth'
        model = models.efficientnet_b2(pretrained=False)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 7)
        )
        model.load_state_dict(torch.load(LOAD_MODEL_PATH))
        model = model.to(device)

    elif model_name == 'regnet_x_8gf':
        LOAD_MODEL_PATH = 'model/regnet_x_8gf_batch4_epoch10_best.pth'
        model = models.regnet_x_8gf(pretrained=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 7)
        model.load_state_dict(torch.load(LOAD_MODEL_PATH))
        model = model.to(device)

    elif model_name == 'inception_v3':
        LOAD_MODEL_PATH = 'model/inception_v3_batch4_epoch50_best.pth'
        model = models.inception_v3(pretrained=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 7)
        model.load_state_dict(torch.load(LOAD_MODEL_PATH))
        model = model.to(device)

    elif model_name == 'regnet_x_16gf':
        LOAD_MODEL_PATH = 'model/regnet_x_16gf_batch4_epoch10_best.pth'
        model = models.regnet_x_16gf(pretrained=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 7)
        model.load_state_dict(torch.load(LOAD_MODEL_PATH))
        model = model.to(device)

    elif model_name == 'regnet_x_32gf':
        LOAD_MODEL_PATH = 'model/regnet_x_32gf_batch4_epoch10_best.pth'
        model = models.regnet_x_32gf(pretrained=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 7)
        model.load_state_dict(torch.load(LOAD_MODEL_PATH))
        model = model.to(device)

    return model


def parse_config():
    """Define parse config
    model: which model you would select
    ensemble: use ensemble or not
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='regnet_x_8gf', type=str)
    parser.add_argument("--ensemble", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    test_transformer = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    inception_transformer = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        normalize
    ])

    args = parse_config()

    if args.ensemble:
        model1 = select_model('regnet_x_8gf')
        model1.eval()
        model2 = select_model('inception_v3')
        model2.eval()
        model3 = select_model('regnet_x_16gf')
        model3.eval()
        model4 = select_model('regnet_x_32gf')
        model4.eval()
        # model5 = select_model('efficientnet_b2')
        # model5.eval()
        # model6 = select_model('regnet')
        # model6.eval()
    else:
        model = select_model(args.model)
        model.eval()

    test_images = os.listdir('crops/test_stg2')
    # test_images = glob('test_stg2/*.jpg')
    # test1_images = os.listdir('test_stg1')
    print(len(test_images))
    test_images.sort()
    submission = pd.read_csv('sample_submission_stg2.csv', index_col='image')
    # submission.loc[:,:] = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    outputs_list = []
    # image order is important to your result
    with torch.no_grad():
        for j, filename in enumerate(tqdm(test_images)):
            img = Image.open(os.path.join('crops/test_stg2', filename)).convert('RGB')
            img_tensor = test_transformer(img)
            img_tensor = img_tensor.unsqueeze(0)
            img_tensor = img_tensor.to(device)
            if args.ensemble:
                output1 = model1(img_tensor)
                output3 = model3(img_tensor)
                output4 = model4(img_tensor)
                # prob = (F.softmax(output3.data, dim=1) + F.softmax(output4.data, dim=1)) / 2
                img_tensor = inception_transformer(img)
                img_tensor = img_tensor.unsqueeze(0)
                img_tensor = img_tensor.to(device)
                output2 = model2(img_tensor)
                prob = (F.softmax(output1.data, dim=1) + F.softmax(output2.data, dim=1)
                + F.softmax(output4.data, dim=1)) / 3
            elif args.model == 'inception_v3':
                img_tensor = inception_transformer(img)
                img_tensor = img_tensor.unsqueeze(0)
                img_tensor = img_tensor.to(device)
                outputs = model(img_tensor)
                prob = F.softmax(outputs.data, dim=1)
            else:
                outputs = model(img_tensor)
                prob = F.softmax(outputs.data, dim=1)
            prob = prob.cpu().numpy().tolist()[0]
            content = []
            content.extend(prob[:4])
            content.append(0.0)
            content.extend(prob[4:])
            # content.extend(prob.cpu().numpy().tolist()[0])
            submission.loc[f'test_stg2/{filename}'] = content

    submission.to_csv(args.output)
