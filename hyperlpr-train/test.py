import cv2
import os
import numpy as np 
from keras.models import load_model 

img_dir = './car_pic/image/val/'
filename = '川010168'
# img = cv2.imread(os.path.join(img_dir, '川S77777.jpg'))
img = cv2.imdecode(np.fromfile(img_dir + '川A0919A.jpg', dtype=np.uint8), -1)
a = np.array(img)
a = a.reshape(128,40,3)
a = a.reshape((1,128,40,3))
model = load_model('model_weight.h5')

predict = model.predict(a)
print ('识别为：')
print (predict)
cv2.imshow('川A0919A.jpg',img)
cv2.waitKey(0)