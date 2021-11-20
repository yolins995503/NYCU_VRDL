NYCU_VRDL_HW2  
  
Using the yolov5 to implement the object detection  
  
Step: Prepare Train Inference  
  
  Prepare :  
    
  1. Install the environment  
     Command : pip install -r requirements.txt  
       
  2. Modify the （get_label.py） to declare the (training images dir)  
     (line 6 , training images dir , relative to 'path')  
     (line 6 , training images dir , relative to 'path')  
       
  3. Execute the (get_label.py) to obtain the training labels in txt (N txt for N images)  
     Command : python get_label.py  
     Input : digitStruct.mat  
     Output : training labels in labels folder
     
  4. Modify the (dataset.yaml) to declare the (training images dir) , (validation images dir) and (number of labels)  
     (line 2 , dataset root dir)  
     (line 3 , train images  , relative to 'path')   
     (line 4 , valid images  , relative to 'path')   
     (line 8 , number of classes)  
     (line 9 , class names)  
      
    
  Train : 
    
  Inference : 
  
Reference : https://github.com/ultralytics/yolov5  
