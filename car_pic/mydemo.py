import sys
#reload(sys)
#sys.setdefaultencoding("utf-8")
#coding=utf-8
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import time
import LPRdemo as pr
import cv2
import numpy as np

# 测试模型检测所需的时间
def SpeedTest(model,image_path):
    grr = cv2.imread(image_path)
    model.SimpleRecognizePlateByE2E(grr) #先运行一遍
    t0 = time.time()
    for _ in range(20):
        model.SimpleRecognizePlateByE2E(grr)
    t = (time.time() - t0)/20.0   #前面循环了20次
    print ("Image size :" + str(grr.shape[1])+"x"+str(grr.shape[0]) +  " need " + str(round(t*1000,2))+"ms")
    return grr
# 画检测框
def drawRectBox(image,rect,addText):
    cv2.rectangle(image, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])), (0,0, 255), 2,cv2.LINE_AA)
    cv2.rectangle(image, (int(rect[0]-1), int(rect[1])-16), (int(rect[0] + 115), int(rect[1])), (0, 0, 255), -1,
                  cv2.LINE_AA)
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    fontC = ImageFont.truetype("./Font/platech.ttf", 14, 0)
    draw.text((int(rect[0]+1), int(rect[1]-16)), addText, (255, 255, 255), font=fontC) #已经是utf-8格式了
    imagex = np.array(img)
    return imagex

if __name__ == "__main__":
    image_path ="./car_pic/0.jpg"
    model = pr.LPR("model/cascade_lbp.xml","model/model12.h5","model/my_model_weights_12.h5")
    grr = SpeedTest(model,image_path)
    # cv2.imshow('test',grr)
    #grr = cv2.imread(image_path)
    
    try:
        for pstr,confidence,rect in model.SimpleRecognizePlateByE2E(grr):
                if confidence>0.8:
                    image = drawRectBox(grr, rect, pstr+" "+str(round(confidence,3)))
                    print ("plate_str:%s"%pstr)
                    print ("plate_confidence:%.3f"%confidence)
        cv2.imshow("image",image)
        cv2.waitKey(0)          #等待键盘输入
        cv2.destroyAllWindows() #关闭所有创建的窗口
    except:
        print("没有识别到车牌，程序退出！")
