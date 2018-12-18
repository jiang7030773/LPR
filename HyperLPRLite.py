#coding=utf-8
import cv2
import time
import numpy as np
from keras import backend as K
from keras.models import *
from keras.layers import *
from keras.utils.vis_utils import plot_model

chars = [u"京", u"沪", u"津", u"渝", u"冀", u"晋", u"蒙", u"辽", u"吉", u"黑", u"苏", u"浙", u"皖", u"闽", u"赣", u"鲁", u"豫", u"鄂", u"湘", u"粤", u"桂",
             u"琼", u"川", u"贵", u"云", u"藏", u"陕", u"甘", u"青", u"宁", u"新", u"0", u"1", u"2", u"3", u"4", u"5", u"6", u"7", u"8", u"9", u"A",
             u"B", u"C", u"D", u"E", u"F", u"G", u"H", u"J", u"K", u"L", u"M", u"N", u"P", u"Q", u"R", u"S", u"T", u"U", u"V", u"W", u"X",
             u"Y", u"Z",u"港",u"学",u"使",u"警",u"澳",u"挂",u"军",u"北",u"南",u"广",u"沈",u"兰",u"成",u"济",u"海",u"民",u"航",u"空"
             ]

class LPR():
    def __init__(self,model_detection,model_finemapping,model_seq_rec):
        self.watch_cascade = cv2.CascadeClassifier(model_detection)
        self.modelFineMapping = self.model_finemapping()
        self.modelFineMapping.load_weights(model_finemapping)
        self.modelSeqRec = self.model_seq_rec(model_seq_rec)

    def computeSafeRegion(self,shape,bounding_rect):
        top = bounding_rect[1] # y
        bottom  = bounding_rect[1] + bounding_rect[3] # y +  h
        left = bounding_rect[0] # x
        right =   bounding_rect[0] + bounding_rect[2] # x +  w
        min_top = 0
        max_bottom = shape[0]
        min_left = 0
        max_right = shape[1]
        if top < min_top:
            top = min_top
        if left < min_left:
            left = min_left
        if bottom > max_bottom:
            bottom = max_bottom
        if right > max_right:
            right = max_right
        return [left,top,right-left,bottom-top]

    def cropImage(self,image,rect):
        x, y, w, h = self.computeSafeRegion(image.shape,rect)
        return image[y:y+h,x:x+w]
    '''
    1.resize图像大小，cv2.resize函数，按照原来图像比例 
    2.裁剪图片，根据输入的top_bottom_padding_rate如果是0.1，那么上面裁剪掉0.1*height，下面也裁剪掉0.1*height 
    3.将图像从rgb转化为灰度 cv2.cvtColor函数，cv2.COLOR_RGB2GRAY 
    4.根据前面的cv2.CascadeClassifier()物体检测模型(3)，输入image_gray灰度图像，边框可识别的最小size，最大size，
      输出得到车牌在图像中的offset，也就是边框左上角坐标( x, y )以及边框高度( h )和宽度( w ) 
    5.对得到的车牌边框的bbox进行扩大，也就是宽度左右各扩大0.14倍，高度上下各扩大0.15倍。 
    6.返回图片中所有识别出来的车牌边框bbox，这个list作为返回结果。
    '''
    def detectPlateRough(self,image_gray,resize_h = 720,en_scale =1.08 ,top_bottom_padding_rate = 0.05):
        if top_bottom_padding_rate>0.2:
            print("error:top_bottom_padding_rate > 0.2:",top_bottom_padding_rate)
            exit(1)
        height = image_gray.shape[0]  #求高度
        padding = int(height*top_bottom_padding_rate) 
        scale = image_gray.shape[1]/float(image_gray.shape[0])
        image = cv2.resize(image_gray, (int(scale*resize_h), resize_h))  #resize_h = image.shape[0]，resize的尺寸没变化？？？
        #cv2.resize()   要注意参数顺序为：宽×高×通道
        image_color_cropped = image[padding:resize_h-padding,0:image_gray.shape[1]] #上下边框裁剪了48像素
        image_gray = cv2.cvtColor(image_color_cropped,cv2.COLOR_RGB2GRAY) #变灰
        #cv2.imshow("image",image)   #by jiang
        watches = self.watch_cascade.detectMultiScale(image_gray, en_scale, 2, minSize=(36, 9),maxSize=(36*40, 9*40))
        cropped_images = []
        for (x, y, w, h) in watches:
            x -= w * 0.14
            w += w * 0.28
            y -= h * 0.15
            h += h * 0.3
            cropped = self.cropImage(image_color_cropped, (int(x), int(y), int(w), int(h)))
            #cv2.imshow("test",cropped)
            cropped_images.append([cropped,[x, y+padding, w, h]])
        return cropped_images
    
    #匹配文字
    def fastdecode(self,y_pred):
        results = ""
        confidence = 0.0
        table_pred = y_pred.reshape(-1, len(chars)+1)
        res = table_pred.argmax(axis=1)
        for i,one in enumerate(res):
            if one<len(chars) and (i==0 or (one!=res[i-1])):
                results+= chars[one]
                confidence+=table_pred[i][one]
        confidence/= len(results)
        return results,confidence

    '''
    model_path为模型weights文件路径 
    ocr部分的网络模型(keras模型) 
    输入层：164*48*3的tensor 
    输出层：长度为7 的tensor，类别有len(chars)+1种
    '''
    def model_seq_rec(self,model_path):
        width, height, n_len, n_class = 164, 48, 7, len(chars)+ 1
        rnn_size = 256
        input_tensor = Input((164, 48, 3))
        x = input_tensor
        base_conv = 32
        for i in range(3):
            x = Conv2D(base_conv * (2 ** (i)), (3, 3))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
        conv_shape = x.get_shape()
        x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(x)
        x = Dense(32)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(x)
        gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(x)
        gru1_merged = add([gru_1, gru_1b])
        gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
        gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)
        x = concatenate([gru_2, gru_2b])
        x = Dropout(0.25)(x)
        x = Dense(n_class, kernel_initializer='he_normal', activation='softmax')(x)
        base_model = Model(inputs=input_tensor, outputs=x)
        base_model.load_weights(model_path)
        # plot_model(base_model,to_file=" gru_model.png") #画出模型 by jiang
        base_model.summary()
        return base_model

    #对车牌的左右边界进行回归 
    def model_finemapping(self):  
        input = Input(shape=[16, 66, 3])  # change this shape to [None,None,3] to enable arbitraty shape input
        x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input)
        x = Activation("relu", name='relu1')(x)
        x = MaxPool2D(pool_size=2)(x)
        x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
        x = Activation("relu", name='relu2')(x)
        x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
        x = Activation("relu", name='relu3')(x)
        x = Flatten()(x)
        output = Dense(2,name = "dense")(x)
        output = Activation("relu", name='relu4')(output)
        model = Model([input], [output])
        #plot_model(model,to_file="cnn_model.png") #画出模型 by jiang
        return model

    '''
    1.将原来车牌图像resize大小：66*16*3 
    2.将原来灰度图颜色通道[0, 255]转化为float类型[0,1] 
    3.将输入66*16(float),输入进模型进行测试self.modelFineMapping.predict
    '''
    def finemappingVertical(self,image,rect):
        resized = cv2.resize(image,(66,16))
        resized = resized.astype(np.float)/255
        res_raw= self.modelFineMapping.predict(np.array([resized]))[0]
        res  =res_raw*image.shape[1]
        res = res.astype(np.int)
        H,T = res
        H-=3
        if H<0:
            H=0
        T+=2
        if T>= image.shape[1]-1:
            T= image.shape[1]-1
        rect[2] -=  rect[2]*(1-res_raw[1] + res_raw[0])
        rect[0]+=res[0]
        image = image[:,H:T+2]
        image = cv2.resize(image, (int(136), int(36)))
        return image,rect
    
    '''
    1.将前面的(136, 36)图像resize成(164, 48) 
    2.将图像转置，输入
    '''
    def recognizeOne(self,src):
        x_tempx = src
        x_temp = cv2.resize(x_tempx,(164,48))
        x_temp = x_temp.transpose(1, 0, 2)
        y_pred = self.modelSeqRec.predict(np.array([x_temp]))
        y_pred = y_pred[:,2:,:]
        return self.fastdecode(y_pred)

    def SimpleRecognizePlateByE2E(self,image):
        images = self.detectPlateRough(image,image.shape[0],top_bottom_padding_rate=0.1)
        res_set = []
        #对于每个车牌区域的for循环中,经过fineMappingVertical处理后输入到recognizeOne函数，进行ocr识别
        for j,plate in enumerate(images):
            plate, rect  = plate #二维数组赋值
            image_rgb,rect_refine = self.finemappingVertical(plate,rect)
            # t0 = time.time()
            res,confidence = self.recognizeOne(image_rgb)
            # tdiff = time.time()-t0
            # print(tdiff)
            res_set.append([res,confidence,rect_refine])
        return res_set
