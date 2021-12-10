NYCU_VRDL_HW3

Using the dectectron2 to implement the segmentation

Step: Prepare Train Inference

Prepare:  
  
1. Prepare the training data for preprocessing:  
   The training folder contrain the original images and mask  
   Train_Folder :
   Image_name_Folder:
   images:  
   masks:  
         
2. Execute the (mask2coco.py) to obtain the traing label in COCO format  
   Command: python maskcoco.py -train_folder /home/bsplab/Documents/yolin/VRDL/dataset/stages1_train  
   Select the training folder path (-train_folder /home/bsplab/Documents/yolin/VRDL/dataset/stages1_train)
     
3. Prepare the training data and testing data for execute detectron2  
   The training images 
            
         
