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
    6.  Ensemble models
    7.  Clip the answer.csv  

 Crop the training data into many training images by using the (label.json):  

    Execute the (crop_dataset.py) to crop training data into many training images  
    Command : python crop_dataset.py  
    Input : label.json * 7 , training data
    Output : training images in each class folder  
   
 Train the classification model by training images:  
   
    Execute the (train_crop.py) to implement the classification  
    Command : python train_crop.py  
    Input : traing images  
    Output : model.pt  
   
 Train the objection detection model by training images:  
     
    1. Clone the source of yolov5  
       Command : git clone https://github.com/ultralytics/yolov5  

    2. Install the environment  
       Command : pip install -r requirements.txt  

    3. Modify the （get_label.py） to declare the (training images dir)  
       (line 6 , training images dir , relative to 'path')  
       (line 6 , training images dir , relative to 'path')  

    4. Execute the (label_mask.py) to obtain the training labels in txt (N txt for N images)  
       Command : python label_mask.py  
       Input : label.json * 7 , training data
       Output : training labels in labels folder  

    5. It is necessary to put (images) and (labels) in the same folder  

    6. Modify the (dataset.yaml) to declare the (training images dir) , (validation images dir) and (number of labels)
       (line 2 , dataset root dir)  
       (line 3 , train images , relative to 'path')  
       (line 4 , valid images , relative to 'path')  
       (line 8 , number of classes)  
       (line 9 , class names)  

     7. Execute the (train.py) to training the model
        Command : python train.py --img 640 --batch 16 --epochs 3 --data dataset.yaml --weights yolov5s.pt  
        Select the images size (--img 640)  
        batch size (--batch 16)  
        epochs (--epochs 3)  
        training data (--data dataset.yaml)  
        pretrain model weight (--weights yolov5s.pt) There are four pretrain model weight (yolov5s.pt , yolov5m.pt , yolov5l.pt , yolov5x.pt)  
        Output : model weight (best.pt , last.pt) and other training record in ( /run/train/exp ) folder  
            
Detect the boundary box of fishes in testing data and crop into testing images:  

    Execute the (detect.py) to crop testing data into testing images  
    Command : python detect.py --source /home/bsplab/Documents/yolin/VRDL_HW2/test/test  
    --weights /home/bsplab/Documents/yolin/VRDL_HW2/yolov5/yolov5-master/runs/train/exp6/weights/best.pt --conf 0.5  
    Select the test images dir (--source /home/bsplab/Documents/yolin/VRDL_HW2/test/test)  
    model weight (--weights /home/bsplab/Documents/yolin/VRDL_HW2/yolov5/yolov5-master/runs/train/exp6/weights/best.pt)  
    threshold of confidence (--conf 0.5)  
    Input : esting images ,objection detection model weight
    Output : testing images in the folder  
          
Predict the class of testing images:
  
    Execute the (inference_crop.py) to obtain the answer.csv
    Command : python inference_crop.py  
    Input :  testing images , classification model weight  
    Output : answer.csv  
  
Ensemble models:
    
    Execute the (ensemble.ipynb) to combine the different model prediction together
      
 Clip the answer.csv:  
      
    Execute the (clip.py) to obtain the new answer.csv  
    Commond : python clip.py 
    Input :  answer.csv  
    Output : answer.csv  
    
Model weight:  
  
    Object detection model:  https://drive.google.com/drive/folders/1vsUzB1L3Ta-KInFPWsIXGl5NOLkPwO_2?usp=sharing (in objection detection folder)  
    Classificatio model:  https://drive.google.com/drive/folders/1vsUzB1L3Ta-KInFPWsIXGl5NOLkPwO_2?usp=sharing (in classification folder)  

Reference : 
    
    [1] https://github.com/ultralytics/yolov5  
    [2] https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/discussion/25428  
    [3] https://github.com/autoliuweijie/Kaggle/tree/master/NCFM/datasets  
