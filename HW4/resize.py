import cv2
import os 
def resize_images(img , height ,width):
    resize_image = cv2.resize(img, (height, width), interpolation=cv2.INTER_AREA)
    return resize_image

if __name__=='__main__':
    input_folder = 'training_hr_images'
    out_folder = 'training_lr_images'
    files= os.listdir(input_folder)
    for idx in range(len(files)):
        input_path = input_folder+'/'+files[idx]
        image = cv2.imread(input_path)
        # print (image.shape[0])
        out_image = resize_images( image , image.shape[1]//3, image.shape[0]//3)
        out_path = out_folder+'/'+files[idx]
        cv2.imwrite(out_path , out_image)