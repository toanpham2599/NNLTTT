import os
import cv2 #Thư viện xử lý ảnh và laod ảnh
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tflearn #Thư viện dùng để training model CNN
from random import shuffle
from tqdm import tqdm
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import *

#ĐẶt thông số learning rate, IMG size đường dẫn với data traning và dadataa testing 
Learningrate = 0.001
TRAIN_DIR = 'D:\\Học Tập\\Ngôn ngữ lập trình tiên tiến\\Đồ án cuối kì\\Nhận diện chó mèo\\dogs-vs-cats\\train'
TEST_DIR = 'D:\\Học Tập\\Ngôn ngữ lập trình tiên tiến\\Đồ án cuối kì\\Nhận diện chó mèo\\dogs-vs-cats\\test'
Img_size = 50
MODEL_NAME = 'Catanddog_Detect'

#Lấy dự liệu dầu vào cho training Model
#Nếu wordlabel sau khi được cát có giá trị là cat thì ta trả về [1,0]
def label(image_name) :
    word_label =image_name[-3]
    if word_label == 'cat' :
        return np.array([1,0])
    elif word_label == 'dog' :
        return np.array([0,1])

#Lấy các ảnh cat/dog, resize ảnh
def create_train_data() :
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)) :
        path = os.path.join(TRAIN_DIR,img)
        #Load ảnh và convert sang dạng ma trận trước đó chuyển hết sang màu xám
        img_data = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        #Risize ảnh
        img_data = cv2.resize(img_data, (Img_size, Img_size))
        training_data.append([np.array(img_data), create_label (img)])
        shuffle(training_data)
        np.save('train_data.npy', training_data)
        return training_data

#Thực hiện create_train_data và tách dữ liệu thành training data và testing data
#Load data
training_data = create_train_data()

#Chia tập dữ liệu thành train và test tư bộ dữ liệu train ban đầu.
#bộ dữ liệu test không có label nên chỉ dùng để test random
from sklearn.model_selection import train_test_splittrain, test = train_test_split(train_data, test_size = 0.2, random_state = 42)

x_train = np.array([i[0] for i in train]).reshape(-1, Img_size, Img_size, 1)
y_train = [i[1] for i in train]
x_test = np.array([i[0] for i in test]).reshape(-1, Img_size, Img_size, 1)
y_test = [i[1] for i in test]

