import h5py
import glob
import cv2 


train_img_list = glob.glob('/home/bsplab/Documents/yolin/VRDL_HW2/train/train/*.png')


hdf5_data = h5py.File("digitStruct.mat",'r')
for idx in range(len(train_img_list)):

	attrs = {}
	item = hdf5_data['digitStruct']['bbox'][idx].item()
	for key in ['label','left','top','width','height']:
		attr = hdf5_data[item][key]
		values = [hdf5_data[attr.value[i].item()].value[0][0]		
					for i in range(len(attr))] if len(attr)>1 else [attr.value[0][0]]
		attrs[key]=values

	
	imgData = cv2.imread('/home/bsplab/Documents/yolin/VRDL_HW2/train/train/' + str(idx + 1) + '.png')

	bboxNum=len(attr)

	height, width, channels = imgData.shape
	imgDataHeight=height
	imgDataWidth=width
	for j in range(bboxNum):
		xCenter = attrs['left'][j] + attrs['width'][j] / 2		
		yCenter = attrs['top'][j] + attrs['height'][j] / 2
		xCenter = xCenter / imgDataWidth		
		yCenter = yCenter / imgDataHeight		
		imgWidth = attrs['width'][j] / imgDataWidth		
		imgHeight = attrs['height'][j] / imgDataHeight	
		
		file = open("label/{}.txt".format(str(idx+1)), 'a')
		file.writelines(str((attrs['label'][j]-1)) + " " + str(xCenter) + " " + str(yCenter) + " " + str(imgWidth) + " " + str(imgHeight) + '\n')
		file.close()