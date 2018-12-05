#-*- coding: utf-8 -*-
import cv2
import sys
import gc
from test_model import build_model

if __name__ == '__main__':
    # if len(sys.argv) != 2:
    #     print("Usage:%s camera_id\r\n" % (sys.argv[0]))
    #     sys.exit(0)
        
    #加载模型
    model = build_model('./model/my_model_weights.h5') 
              
    #框住人脸的矩形边框颜色       
    color = (0, 255, 0)
    
    #捕获指定摄像头的实时视频流
    cv2.namedWindow('License plate recognition')
    cap = cv2.VideoCapture(0)
    
    #人脸识别分类器本地存储路径
    cascade_path = "./model/cascade_lbp.xml"    
    
    #循环检测识别人脸
    while True:
        _, frame = cap.read()   #读取一帧视频
        
        #图像灰化，降低计算复杂度
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #使用人脸识别分类器，读入分类器
        cascade = cv2.CascadeClassifier(cascade_path)                

        #利用分类器识别出哪个区域为人脸
        faceRects = cascade.detectMultiScale(frame_gray, scaleFactor = 1.2, minNeighbors = 3, minSize = (128, 40))        
        # faceRects = cascade.detectMultiScale(frame_gray, en_scale=1.08, 2, minSize=(128, 40),maxSize=(128, 40))

        if len(faceRects) > 0:                 
            for faceRect in faceRects: 
                x, y, w, h = faceRect
                
                #截取脸部图像提交给模型识别这是谁
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                faceID = model.predict(image)   
                
                #如果是“我”
                if faceID == 0:                                                        
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)
                    
                    #文字提示是谁
                    cv2.putText(frame,'Dady', 
                                (x + 30, y + 30),                      #坐标
                                cv2.FONT_HERSHEY_SIMPLEX,              #字体
                                1,                                     #字号
                                (255,0,255),                           #颜色
                                2)                                     #字的线宽
                else:
                    pass
                            
        cv2.imshow("识别PR", frame)
        
        #等待10毫秒看是否有按键输入
        k = cv2.waitKey(10)
        #如果输入q则退出循环
        if k & 0xFF == ord('q'):
            break

    #释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()