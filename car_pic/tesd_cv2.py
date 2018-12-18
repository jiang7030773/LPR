import cv2
import time
import sys
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import numpy as np 
import HyperLPRLite as pr

def SpeedTest(image_path):
    grr = cv2.imread(image_path)
    model = pr.LPR("model/cascade.xml", "model/model12.h5", "model/ocr_plate_all_gru.h5")
    model.SimpleRecognizePlateByE2E(grr)
    t0 = time.time()
    for _ in range(20):
        model.SimpleRecognizePlateByE2E(grr)
    t = (time.time() - t0)/20.0   #前面循环了20次
    print ("Image size :" + str(grr.shape[1])+"x"+str(grr.shape[0]) +  " need " + str(round(t*1000,2))+"ms")
    return grr

def drawRectBox(image,rect,addText):
    cv2.rectangle(image, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])), (0,0, 255), 2,cv2.LINE_AA)
    cv2.rectangle(image, (int(rect[0]-1), int(rect[1])-16), (int(rect[0] + 115), int(rect[1])), (0, 0, 255), -1,
                  cv2.LINE_AA)
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    fontC = ImageFont.truetype("./Font/platech.ttf", 14, 0)
    # draw.text((int(rect[0]+1), int(rect[1]-16)), addText.encode("utf-8"), (255, 255, 255), font=fontC)
    draw.text((int(rect[0]+1), int(rect[1]-16)), addText, (255, 255, 255), font=fontC) #已经是utf-8格式了
    imagex = np.array(img)
    return imagex


cap = cv2.VideoCapture(0) #打开摄像头
# cv2.namedWindow('LPR',cv2.WINDOW_NORMAL)       
font = cv2.FONT_HERSHEY_SIMPLEX
ret = cap.set(3,640)
ret = cap.set(4,480) 


model = pr.LPR("model/cascade_lbp.xml","model/model12.h5","model/ocr_plate_all_gru.h5")
# 计数帧数q
framecount = 0

while(1):
    # get a frame
    ret, frame = cap.read()
    # frame = cv2.flip(frame, 3)
    frame = cv2.resize(frame, (640,480))
    # 计数帧数
    if (framecount%3 != 0):
        framecount += 1
        continue

    if ret == True:
        print('成功读取图片')
    if model.SimpleRecognizePlateByE2E(frame) == []:
        print('no lpr')
        # cv2.putText(frame,'No License plate',(10,20),font,0.7,(255,0,0),2,1)
        image = drawRectBox(frame, [320,240,100,30], 'No License plate')
        cv2.imshow("capture", image)
        continue
    for pstr,confidence,rect in model.SimpleRecognizePlateByE2E(frame):
        if confidence>0.8:
            image = drawRectBox(frame, rect, pstr+" "+str(round(confidence,3)))
            print ("plate_str:%s"%pstr)
            print ("plate_confidence:%.3f"%confidence)
            cv2.imshow("capture", image)     #生成摄像头窗口
        else:
            continue

    if cv2.waitKey(1) & 0xFF == ord('q'):   #如果按下q 就截图保存并退出
        cv2.imwrite("./car_pic/test.png", image)   #保存路径
        break
 
cap.release()
cv2.destroyAllWindows()
