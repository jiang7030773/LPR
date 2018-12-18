# -*- coding:utf-8 -*-
import os
import cv2
import numpy as np
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from plateCommon import *

class GenPlate:
    def __init__(self,fontCh,fontEng,NoPlates):
        self.fontC =  ImageFont.truetype(fontCh,43,0)
        self.fontE =  ImageFont.truetype(fontEng,60,0)
        self.img=np.array(Image.new("RGB", (226,70),(255,255,255)))
        self.bg  = cv2.resize(cv2.imread("./images/template.bmp"),(226,70))
        self.smu = cv2.imread("./images/smu2.jpg")
        self.noplates_path = []
        for parent,parent_folder,filenames in os.walk(NoPlates):
            for filename in filenames:
                path = parent+"/"+filename
                self.noplates_path.append(path)


    def draw(self,val):
        offset= 2
        self.img[0:70,offset+8:offset+8+23]= GenCh(self.fontC,val[0])
        self.img[0:70,offset+8+23+6:offset+8+23+6+23]= GenCh1(self.fontE,val[1])
        for i in range(5):
            base = offset+8+23+6+23+17 +i*23 + i*6
            self.img[0:70, base  : base+23]= GenCh1(self.fontE,val[i+2])
        return self.img
    def generate(self,text):
        if len(text) == 7:
            fg = self.draw(text)
            fg = cv2.bitwise_not(fg)
            com = cv2.bitwise_or(fg,self.bg)
            # com = rot(com,r(60)-30,com.shape,30)
            com = rot(com,r(40)-20,com.shape,20)
            com = rotRandrom(com,10,(com.shape[1],com.shape[0]))
            com = AddSmudginess(com,self.smu)

            # com = tfactor(com)
            com = random_envirment(com,self.noplates_path)
            com = AddGauss(com, 1+r(2))
            com = addNoise(com)
            return com
    def genPlateString(self):
        plateStr = ""
        box = [0,0,0,0,0,0,0]
        for cpos in range(len(box)):
            if cpos == 0:
                plateStr += chars[np.random.randint(0,31)]
            elif cpos == 1:
                plateStr += chars[np.random.randint(41,64)]
            else:
                plateStr += chars[np.random.randint(31,64)]
               
        return plateStr

    def genBatch(self, batchSize,outputPath,size):
        # AplateStr = []
        # Aimg = []
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
        for i in range(batchSize):
            plateStr = self.genPlateString()
            img = self.generate(plateStr)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img,size)
            filename = os.path.join(outputPath, plateStr + ".jpg")
            # filename = os.path.join(outputPath, str(i+start).zfill(5) + '_' + plateStr + ".jpg")
            cv2.imencode('.jpg', img)[1].tofile(filename)
            # AplateStr.append(plateStr)
            # Aimg.append(img)
        # return   AplateStr, Aimg


G = GenPlate("./font/platech.ttf",'./font/platechar.ttf',"./NoPlates")
G.genBatch(5000,"data",(128,40))
