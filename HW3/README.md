NYCU_VRDL_HW3
# NYCU_VRDL_HW3
### Using the dectectron2 to implement the segmentation

It is convenient to execute the (Detectron2_yolin.ipynb) for training and infernece
Training output: (model_final.pth)
Inference Ouput: (answer.json)

## Install Environment
Clone the https://github.com/jsbroks/imantics.git to install the imantics
```
git clone https://github.com/jsbroks/imantics.git 
```
Install
```
 pip install .  
```

## Prepare the data
1. Prepare the training data for preprocessing:
    The training folder contrain the original images and masks  
    Ex:  
    /Train_Folder/Image_name_Folder/images/001.png  
   /Train_Folder/Image_name_Folder/masks/001_mask.png   
```
```
2. Execute the (mask2coco.py) to obtain the traing label in COCO format  
    Output: (train_correct.json)  
```
python maskcoco.py -train_folder /home/bsplab/Documents/yolin/VRDL/dataset/stages1_train
```
3. Prepare the training data and testing data for execute detectron2
    The training images and testing images should be in the folder  
    Ex:  
    /Train_data/001.png  
    /Test_data/100.png  
```
```

## Train
Execute the (train.py) to train the model  
Input: (train_correct.json)  and training images folder  
Output: (model_final.pth)  
```
python train.py  
```

## Inference
Execute the (inference.py) to produce the prediction file  
Input: (model_final.pth) and testing images folder  
Output: (answer.json)  
```
 python inference.py
```

## Model weight link

https://drive.google.com/drive/folders/1kq17ZeJ3vE1YPwW8PXTCluNIGGMonYUR?usp=sharing

## Reference
[1] https://github.com/facebookresearch/detectron2  
  
[2]https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools  

[3] https://github.com/jsbroks/imantics.git