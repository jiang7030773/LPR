from hyperlpr_py3 import pipline as pp
from hyperlpr_py3 import colourDetection
import time 
import cv2
import numpy as np

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


image = cv2.imread("./car_pic/PR/car.jpg")
'''t1 = time.time()
#image,res  = pp.SimpleRecognizePlate(image)
print(colourDetection.judge_plate_color(image))
t = (time.time()-t1)*1000
print("检测一张图片所需要的时间是 %.2f(ms)"%t)
#print(res)
'''

image = rotate_bound(image,30)
cv2.imshow('jiang',image)


# print(image.shape)
# s = cv2.resize(image,(64,48))
# print(s.shape)


# cv2.imshow("",s)
cv2.waitKey(0)          #等待键盘输入
