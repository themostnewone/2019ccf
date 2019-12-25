#数据路径，sub路径，存放csv路径，存放bin路径
data_dir = 'testing/Test_Data/'  # 测试集图片路径
subpath = 'testing/submission_template.csv'  # 测试集csv路径
cunpath = 'blendsub/' #存放csv文件路径
mo=1#1-4

from PIL import Image,ImageEnhance

def rui(image,sharpness):
    image=Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) #opencv格式转为PIL格式
    enh_sha = ImageEnhance.Sharpness(image)
    image = enh_sha.enhance(sharpness)
    image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)  #PIL-opencv
    return image

def se(image,color ):
    image=Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) #opencv格式转为PIL格式
    enh_col = ImageEnhance.Color(image)
    image = enh_col.enhance(color)
    image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)  #PIL-opencv
    return image

def dl(src1,a,g): #对比度，亮度比例
    h, w, ch = src1.shape
    src2 = np.zeros([h, w, ch], src1.dtype)
    dst = cv2.addWeighted(src1, a, src2, 1 - a, g)
    return dst


def rotate(image, angle, scale):
    w = image.shape[1]
    h = image.shape[0]
    # rotate matrix
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    # rotate
    image = cv2.warpAffine(image, M, (w, h))
    return image

def flip(image):
    image = np.fliplr(image)
    return image

def transf(x,model):
    x = cv2.resize(x, (112, 112))
    x = np.transpose(x, (2, 0, 1))
    return  model.get_feature(x)

from sklearn.preprocessing  import normalize
def get_featurs(model, test_list,data_dir):
    cnt = 0
    features = None
    for i, img_path in enumerate(test_list):
        image = cv2.imread(data_dir+img_path)
        fan=flip(image)
        jia1=rotate(image, -5, 1)
        jia2=rotate(image, 5, 1)
        jia3 = rotate(fan, -5, 1)
        jia4 = rotate(fan, 5, 1)
        jia13=rui(image,1.2)
        jia14 = rui(image,0.8)
        jia15 = rui(fan, 1.2)
        jia16 = rui(fan, 0.8)

        jia17 = cv2.resize(image, (122, 144))[16:128, 16:128]  # 122,144
        jia17=transf(jia17, model)

        jia18 = cv2.resize(fan, (122, 144))[16:128, 16:128]  # 122,144
        jia18 = transf(jia18,model)


        image=transf(image, model)
        fan=transf(fan, model)
        jia1=transf(jia1, model)
        jia2 = transf(jia2, model)
        jia3 = transf(jia3, model)
        jia4 = transf(jia4, model)
        jia13 = transf(jia13, model)
        jia14 = transf(jia14, model)
        jia15 = transf(jia15, model)
        jia16 = transf(jia16, model)

        image=image+fan+jia1+jia2+jia3+jia4+jia13+jia14+jia15+jia16+jia17+jia18

        feature = normalize( image.reshape(1, -1), norm='l2')
        cnt += 1

        if cnt%1000==0:
            print(cnt)


        if features is None:
            features = feature
        else:
            features = np.vstack((features, feature))


    return features, cnt


def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
        # key = each.split('/')[1]
        fe_dict[each] = features[i]
    return fe_dict


def cosin_metric(x1, x2):
    x1=x1.astype(float)
    x2=x2.astype(float)
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


class item:

    def __init__(self):
        self.image_size = '112,112'
        self.model = '../input/ccfantfacenetmodel/model-r50/model,0'
        self.ga_model = ''
        self.gpu = 0
        self.det = 0
        self.flip = 0
        self.threshold = 1.24






import face_model
import os
import cv2
import numpy as np
import time
import scipy.io as sio
from collections import OrderedDict
from tqdm import tqdm
import argparse
import torch
from scipy.spatial.distance import pdist



if mo==1:
    modelpath = 'model-r100/model,0'
    shu=1
if mo==2:
    modelpath = 'model-r50/model,0'
    shu=3
if mo == 3:
    modelpath = 'model-r34/model,0'
    shu=4
if mo == 4:
    modelpath = 'model-m/model,0'
    shu=5



args = item()
args.image_size = '112,112'
args.model = modelpath
args.ga_model = ''
args.gpu = 0
args.det = 0
args.flip = 0
args.threshold = 1.24
print(args)
model = face_model.FaceModel(args)

name_list = [name for name in os.listdir(data_dir)]

s = time.time()

face_features, cnt = get_featurs(model, name_list,data_dir)
print('完成特征')
t = time.time() - s
print('total time is {}, average time is {}'.format(t, t / cnt))

print(len(face_features))
print(len(face_features[0]))


fe_dict = get_feature_dict(name_list, face_features)
print('Output number:', len(fe_dict))
sio.savemat('face_embedding_test.mat', fe_dict)

face_features = sio.loadmat('face_embedding_test.mat')

print('Loaded mat')
sample_sub = open(subpath, 'r')  # sample submission file dir
sub = open(cunpath+str(shu)+'.csv', 'w')
print('Loaded CSV')
lines = sample_sub.readlines()
# pbar = tqdm(total=len(lines))
for line in lines:
    pair = line.split(',')[0]
    sub.write(pair + ',')
    a, b = pair.split(':')
    score = '%f' % cosin_metric(face_features[a][0], face_features[b][0])
    sub.write(score + '\n')
    # pbar.update(1)
sample_sub.close()
