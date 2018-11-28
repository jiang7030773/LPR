import cv2
import os
import numpy as np 

img_dir = './car_pic/image/train/'
filename = '川010168'
# img = cv2.imread(os.path.join(img_dir, '川S77777.jpg'))
img = cv2.imdecode(np.fromfile(img_dir + '川S77777\n'.strip()+'.jpg', dtype=np.uint8), -1)
print(img)