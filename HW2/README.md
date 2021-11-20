NYCU_VRDL_HW2  
Use the yolov5 to implement the object detection  
  
Step:(Prepare , Train , Inference)  
  
Prepare:  

1. Instll the environment  
   Command : pip install -r requirements.txt  
     
2. Execute the (get_label.py) to obtain the training label which save  in .txt (N images for N .txt)  in labels folder
   Modify the path in (get_label.py) at line 6 and line 21  
   Command : python get_label.py  
   
   Input : digitStruct.mat  
   Output : txt file in labels folder  
    
3. Need to put the training data(images) and training data (labels) in the same folder  
  
Train:  

1. Modify the path  at line 2 , 3 , 4 and label at line 8 , 9 in (dataset.yaml). 
   path : training images file  
   
2. Exxcute the (train.py) to train and save the model  
   Command : python train.py --img 640 --batch 16 --epochs 3 --data dataset.yaml --weights yolov5s.pt  
   
   Select the images size (--img 640)  
              batch size (--batch 16)  
              epoch (--epochs 3)  
              data (--data dataset.yam)  
              prtrain model weight (--weights yolov5s.pt) There are four wieght that can be select (yolov5s.pt , yolov5m.pt , yolov5l.pt , yolov5x.pt)  
  
  Output :  The model weight and other training data will be stored in the /runs/train/ folder
                
Inference:  

 
1. Execute the (detect.py) to obtain the answer.json  
   Command : python detect.py --source /home/bsplab/Documents/yolin/VRDL_HW2/test/test  
   --weights /home/bsplab/Documents/yolin/VRDL_HW2/yolov5/yolov5-master/runs/train/exp6/weights/best.pt --conf 0.5  
   
   Select the testing images file (--source /home/bsplab/Documents/yolin/VRDL_HW2/test/test)  
              model weights (-weights /home/bsplab/Documents/yolin/VRDL_HW2/yolov5/yolov5-master/runs/train/exp6/weights/best.pt)  
              confidence threshold (--conf 0.5)  

    Output: answer.json  
            The visualization will be stored in the /runs/detect/ folder   
    Output: answer.json  
    
Reference:  
1. https://github.com/ultralytics/yolov5  
