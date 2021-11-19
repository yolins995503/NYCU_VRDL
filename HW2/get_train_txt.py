import h5py
import glob
import cv2 


train_img_list = glob.glob('/home/bsplab/Documents/yolin/VRDL_HW2/train/train/*.png')




for idx in range(len(train_img_list)):



	file = open("train1.txt", 'a')
	file.writelines("/home/bsplab/Documents/yolin/VRDL_HW2/train/train/"+str(idx+1)+".png"+'\n')
	file.close()
	
	