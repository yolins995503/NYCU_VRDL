NYCU_VRDL_Final  
  
Topic : The Nature Conservancy Fisheries Monitoring  
Link : https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring   
  
Using classification model and objection detection model to distinguish the fishes  
  
Procedure :   
1.  Crop the training data into many training images by using the (label.json)  
2.  Train the classification model by training images  
3.  Train the objection detection model by training images  
4.  Detect the boundary box of fishes in testing data and crop into testing images  
5.  Predict the class of testing images  

 Crop the training data into many training images by using the (label.json):  
 
 Execute the (crop_dataset.py) to crop training data into many training images  
 Command : python crop_dataset.py  
 Input : label.json * 7 , training data
 Output : training images in each class folder  
   
 Train the classification model by training images:  
   
 Execute the () 
 Command :  
 Input : 
 Output :  
   
 Train the objection detection model by training images:  
 
 1. Clone the source of yolov5  
    Command : git clone https://github.com/ultralytics/yolov5  
      
 2. Install the environment  
    Command : pip install -r requirements.txt  
 


