NYCU_VRDL_HW2  
  
Using the yolov5 to implement the object detection  
  
Step: Prepare Train Inference  
  
  Prepare :  
    
  1. Install the environment  
     Command : pip install -r requirements.txt  
       
  2. Execute the (get_label.py) to obtain the training labels in txt (N txt for N images)  
     Modify the path of training images folder (line 6 , 21 of get_label.py)  
     Command : python get_label.py  
     Input : digitStruct.mat  
     Output : training labels in labels folder
     
  3. Modify the (dataset.yaml) to declare the training dir , validation data and number of labels
    
  Train : 
    
  Inference : 
  
Reference : https://github.com/ultralytics/yolov5  
