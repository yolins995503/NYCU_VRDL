
NYCU_VRDL_HW3

Using the dectectron2 to implement the segmentation

Step: Prepare Train Inference
  
Prepare:

    Prepare the training data for preprocessing:
    The training folder contrain the original images and masks  
    Ex:  
    /Train_Folder/Image_name_Folder/images/001.png  
    /Train_Folder/Image_name_Folder/masks/001_mask.png  
    
    Execute the (mask2coco.py) to obtain the traing label in COCO format
    Command: python maskcoco.py -train_folder /home/bsplab/Documents/yolin/VRDL/dataset/stages1_train
    Select the training folder path (-train_folder /home/bsplab/Documents/yolin/VRDL/dataset/stages1_train)  
    Output: (train_correct.json)    
      
    Prepare the training data and testing data for execute detectron2
    The training images and testing images should be in the folder  
    Ex:  
    /Train_data/001.png
    /Test_data/100.png  
      
Train:
      
    Execute the (train.py) to train the model  
    Command: python train.py  
    Input: (train_correct.json)  and training images folder  
    Output: (model_final.pth)  
      
Inference:  
  
    Execute the (inference.py) to produce the prediction file  
    Command: python inference.py
    Input: (model_final.pth) and testing images folder
    Output: answer.jsom
  
Reference:
      
    [1] https://github.com/facebookresearch/detectron2  
      
    [2] https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools  
