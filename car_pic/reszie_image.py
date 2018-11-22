import os
import cv2
import sys
import numpy as np
 
def resize_image(path):
    image_nums = 0
    for filename in os.listdir(path):
        if os.path.splitext(filename)[1] == '.jpg':
            img = cv2.imdecode(np.fromfile(path + filename, dtype=np.uint8), -1)
            print(img.shape)
            newImg = cv2.resize(img,dsize=(128,39))
            # newImg = cv2.resize(img, dsize=(int((np.shape(img)[1])*0.2), int((np.shape(img)[0])*0.2)))
            
            # cv2.namedWindow("Image")
            # cv2.imshow("Image",newImg)
            # cv2.waitKey(0)
            #cv2.imwrite(path + filename,newImg)不能保存中文名
            
            cv2.imencode('.jpg', newImg)[1].tofile(path + filename)
            image_nums += 1
            print('成功修改第%d张图片大小'%image_nums)



if __name__ == '__main__':
    path = "./car_pic/3922/"
    resize_image(path)