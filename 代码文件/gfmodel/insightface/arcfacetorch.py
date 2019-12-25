# !pip install tensorflow==1.14.0
# !pip install tensorflow-gpu==1.14.0
# !pip install insightface
import os
import cv2
import numpy as np
import time
import scipy.io as sio
from collections import OrderedDict
from tqdm import tqdm_notebook as tqdm
import insightface

def load_image(img_path, filp=False):
     image = cv2.imread(img_path, 1)
     image = image[-96:,:,:]
     image = cv2.resize(image,(112,112))
     if image is None:
        return None
     if filp:
        image = cv2.flip(image,1,dst=None)
     return image


model = insightface.model_zoo.get_model('arcface_r100_v1')
model.prepare(ctx_id = 0)

# _models = {
#     'arcface_r100_v1': arcface_r100_v1,
#     #'arcface_mfn_v1': arcface_mfn_v1,
#     #'arcface_outofreach_v1': arcface_outofreach_v1,
#     'retinaface_r50_v1': retinaface_r50_v1,
#     'retinaface_mnet025_v1': retinaface_mnet025_v1,
#     'retinaface_mnet025_v2': retinaface_mnet025_v2,
#     'genderage_v1': genderage_v1,
# }

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
    h, w, ch = src1.shape  # 获取shape的数值，height和width、通道
    # 新建全零图片数组src2,将height和width，类型设置为原图片的通道类型(色素全为零，输出为全黑图片)
    src2 = np.zeros([h, w, ch], src1.dtype)
    dst = cv2.addWeighted(src1, a, src2, 1 - a, g)  # addWeighted函数说明如下
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
    # if random_flip and np.random.choice([True, False]):
    image = np.fliplr(image)
    return image


from sklearn.preprocessing  import normalize
def get_featurs(model, test_list,data_dir):
    mp=dict()
    cnt = 0
    #     pbar = tqdm(len(test_list))
    features = None
    for i, img_path in enumerate(test_list):
        image = load_image(data_dir+img_path)

        fan = flip(image)
        jia1 = rotate(image, -5, 1)
        jia2 = rotate(image, 5, 1)
        jia3 = rotate(fan, -5, 1)
        jia4 = rotate(fan, 5, 1)
        jia13 = rui(image, 1.2)
        jia14 = rui(image, 0.8)
        jia15 = rui(fan, 1.2)
        jia16 = rui(fan, 0.8)

        jia17 = cv2.resize(image, (122, 144))[16:128, 16:128]  # 122,144
        jia17 = cv2.resize(jia17, (112, 112))
        jia17 = model.get_embedding(jia17)

        jia18 = cv2.resize(fan, (122, 144))[16:128, 16:128]  # 122,144
        jia18 = cv2.resize(jia18, (112, 112))
        jia18 = model.get_embedding(jia18)


        image=model.get_embedding(image) #还是112,112
        fan=model.get_embedding(fan)
        jia1=model.get_embedding(jia1)
        jia2 = model.get_embedding(jia2)
        jia3 = model.get_embedding(jia3)
        jia4 = model.get_embedding(jia4)
        jia13 = model.get_embedding(jia13)
        jia14 = model.get_embedding(jia14)
        jia15 = model.get_embedding(jia15)
        jia16 = model.get_embedding(jia16)
        image = image + fan + jia1 + jia2 + jia3 + jia4 + jia13 + jia14 + jia15 + jia16+jia17+jia18
        feature = normalize(image.reshape(1, -1), norm='l2')  # [0]
        cnt += 1

        if cnt%1000==0:
            print(cnt)

        if features is None:
            features = feature
        else:
            features = np.vstack((features, feature))
    #         pbar.update(1)

    return features , cnt

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


def produce(data_dir,subpath,mingpath,cunpath): #数据路径，sub路径，存放csv路径，存放bin路径


    name_list = [name for name in os.listdir(data_dir)]
    # name_list = name_list[:10]
    np.save(mingpath+'ming2.npy', np.array(name_list))
    # name_list = np.load('ming2.npy')
    s = time.time()
    face_features, cnt = get_featurs(model, name_list,data_dir)#从这里开始直接copy即可，然后修改函数
    print('完成特征')
    t = time.time() - s
    print('total time is {}, average time is {}'.format(t, t / cnt))

    print(len(face_features))
    print(len(face_features[0]))

    ming = 'face_features2'
    face_features.tofile(mingpath + ming + '.bin')

    fe_dict = get_feature_dict(name_list, face_features)
    print('Output number:', len(fe_dict))
    sio.savemat('gfmodel/insightface/face_embedding_test.mat', fe_dict)
    face_features = sio.loadmat('gfmodel/insightface/face_embedding_test.mat')

    print('Loaded mat')
    sample_sub = open(subpath, 'r')  # sample submission file dir
    sub = open(cunpath+'11.csv', 'w')
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