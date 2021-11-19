NYCU_VRDL_HW2  
Use the yolov5 to implement the object detection  
  
Step:  
  
Prepare:  
1. Instll the environment  
   Command : pip install -r requirements.txt  
     
2. Execute the (get_label.py) to obtain the training label which save  in .txt (N images for N .txt)  in labels folder
   Modify the path in (get_label.py) at line 6 and line 21  
   Command : python get_label.py
   Input : digitStruct.mat  
   Output : txt file in labels folder  
  
Train:  
1. Modify the path  at line 2 , 3 , 4 and label at line 8 , 9 in (dataset.yaml). 
   path : training images file  
   
2. Excute the (train.py) to train and save the model  
   Command : python train.py --img 640 --batch 16 --epochs 3 --data dataset.yaml --weights yolov5s.pt
   Select the images size (--img 640)  
              batch size (--batch 16)  
              epoch (--epochs 3)  
              data (--data dataset.yam)  
              prtrain model weight (--weights yolov5s.pt) There are four wieght that can be select (yolov5s.pt , yolov5m.pt , yolov5l.pt , yolov5s.ptx)
Inference: 
