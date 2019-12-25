import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout, MaxPool2d, \
    AdaptiveAvgPool2d, Sequential, Module
from collections import namedtuple
import time
from PIL import Image, ImageEnhance

#数据路径，sub路径，存放csv路径，存放bin路径
data_dir = 'testing/Test_Data/'  # 测试集图片路径
subpath = 'testing/submission_template.csv'  # 测试集csv路径
cunpath = 'blendsub/' #存放csv文件路径
mo=1#1-6

model_path = 'lvmodel/'

# Support: ['IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False)

        nn.init.xavier_uniform_(self.fc1.weight.data)

        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False)

        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x


class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False), BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]

    return blocks


class Backbone(Module):
    def __init__(self, input_size, num_layers, mode='ir'):
        super(Backbone, self).__init__()
        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        if input_size[0] == 112:
            self.output_layer = Sequential(BatchNorm2d(512),
                                           Dropout(),
                                           Flatten(),
                                           Linear(512 * 7 * 7, 512),
                                           BatchNorm1d(512))
        else:
            self.output_layer = Sequential(BatchNorm2d(512),
                                           Dropout(),
                                           Flatten(),
                                           Linear(512 * 14 * 14, 512),
                                           BatchNorm1d(512))

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)

        self._initialize_weights()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


def IR_50(input_size):
    """Constructs a ir-50 model.
    """
    model = Backbone(input_size, 50, 'ir')

    return model


def IR_101(input_size):
    """Constructs a ir-101 model.
    """
    model = Backbone(input_size, 100, 'ir')

    return model


def IR_152(input_size):
    """Constructs a ir-152 model.
    """
    model = Backbone(input_size, 152, 'ir')

    return model


def IR_SE_50(input_size):
    """Constructs a ir_se-50 model.
    """
    model = Backbone(input_size, 50, 'ir_se')

    return model


def IR_SE_101(input_size):
    """Constructs a ir_se-101 model.
    """
    model = Backbone(input_size, 100, 'ir_se')

    return model


def IR_SE_152(input_size):
    """Constructs a ir_se-152 model.
    """
    model = Backbone(input_size, 152, 'ir_se')

    return model


# Helper function for extracting features from pre-trained models
import torch
import cv2
import numpy as np
import os

import matplotlib.pyplot as plt


def rui(image, sharpness):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # opencv格式转为PIL格式
    enh_sha = ImageEnhance.Sharpness(image)
    image = enh_sha.enhance(sharpness)
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)  # PIL-opencv
    return image


def se(image, color):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # opencv格式转为PIL格式
    enh_col = ImageEnhance.Color(image)
    image = enh_col.enhance(color)
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)  # PIL-opencv
    return image


def dl(src1, a, g):  # 对比度，亮度比例
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
    # if random_flip and np.random.choice([True, False]):
    image = np.fliplr(image)
    return image

def yi(image,a,b):
    rows, cols, _ = image.shape
    M = np.float32([[1, 0, a], [0, 1, b]])
    image = cv2.warpAffine(image, M, (cols, rows))
    return image

def toushi(image,flag):
    rows, cols, ch = image.shape
    pts1 = np.float32([[0, 0], [112, 0], [0, 112], [112, 112]])
    if flag==1:
        pts2 = np.float32([[6, 6], [112, 0], [0, 112], [112, 112]])
    elif flag==2:
        pts2 = np.float32([[0, 0], [106, 6], [0, 112], [112, 112]])
    elif flag==3:
        pts2 = np.float32([[0, 0], [112, 0], [6, 106], [112, 112]])
    elif flag==4:
        pts2 = np.float32([[0, 0], [112, 0], [0, 112], [106, 106]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    image = cv2.warpPerspective(image, M, (cols, rows))
    return image

def transf(img):
    # resize image to [128, 128]
    resized = cv2.resize(img, (112, 112))

    ccropped = resized[..., ::-1]  # BGR to RGB

    # load numpy to tensor
    ccropped = ccropped.swapaxes(1, 2).swapaxes(0, 1)
    ccropped = np.reshape(ccropped, [1, 3, 112, 112])
    ccropped = np.array(ccropped, dtype=np.float32)
    ccropped = (ccropped - 127.5) / 128.0
    ccropped = torch.from_numpy(ccropped)

    return ccropped


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


def extract_feature(test_list, backbone,
                    device, tta,data_dir ):

    cnt = 0
    features = None
    for img_path in test_list:

        # load image
        image = cv2.imread(data_dir + img_path)
        fan = flip(image)
        jia1 = rotate(image, -5, 1)
        jia1 = transf(jia1)
        jia2 = rotate(image, 5, 1)
        jia2 = transf(jia2)
        jia3 = rotate(fan, -5, 1)
        jia3 = transf(jia3)
        jia4 = rotate(fan, 5, 1)
        jia4 = transf(jia4)
        jia13 = rui(image, 1.2)
        jia13 = transf(jia13)
        jia14 = rui(image, 0.8)
        jia14 = transf(jia14)
        jia15 = rui(fan, 1.2)
        jia15 = transf(jia15)
        jia16 = rui(fan, 0.8)
        jia16 = transf(jia16)

        jia17=  cv2.resize(image, (122, 144))[16:128, 16:128]  # 122,144
        jia17 = transf(jia17)

        jia18 = cv2.resize(fan, (122, 144))[16:128, 16:128]  # 122,144
        jia18 = transf(jia18)


        image = transf(image)
        fan = transf(fan)


        with torch.no_grad():
            if tta:
                emb_batch = backbone(image.to(device)).cpu() + backbone(fan.to(device)).cpu() + backbone(
                    jia1.to(device)).cpu() + backbone(jia2.to(device)).cpu() + backbone(
                    jia3.to(device)).cpu() + backbone(jia4.to(device)).cpu() + backbone(
                    jia13.to(device)).cpu() + backbone(jia14.to(device)).cpu() + backbone(
                    jia15.to(device)).cpu() + backbone(jia16.to(device)).cpu()  + backbone(jia17.to(device)).cpu()  + backbone(jia18.to(device)).cpu()   # 在这里加特征
                feature = l2_norm(emb_batch)
            else:
                feature = l2_norm(backbone(ccropped.to(device)).cpu())

        cnt += 1

        if cnt % 1000 == 0:
            print(cnt)


        if features is None:
            features = feature
        else:
            features = np.vstack((features, feature))

    return  features , cnt



def cosin_metric(x1, x2):
    x1 = x1.astype(float)
    x2 = x2.astype(float)
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
        # key = each.split('/')[1]
        fe_dict[each] = features[i]
    return fe_dict


import scipy.io as sio


if mo==1:
    model_root = model_path + 'backbone_ir50_asia.pth'
    backbone = IR_50([112, 112])
    shu=5

if mo==2:
    model_root=model_path+'backbone_ir50_ms1m_epoch63.pth'
    backbone = IR_50([112, 112])
    shu=6

if mo==3:
    model_root=model_path+'backbone_ir50_ms1m_epoch120.pth'
    backbone = IR_50([112, 112])
    shu=7

if mo==4:
    model_root = model_path + 'Backbone_IR_152_Epoch_37_Batch_841528_Time_2019-06-06-02-06_checkpoint.pth'
    backbone = IR_152([112, 112])
    shu=8

if mo == 5:
    model_root=model_path+'Backbone_IR_152_Epoch_59_Batch_1341896_Time_2019-06-14-06-04_checkpoint.pth'
    backbone = IR_152([112, 112])
    shu=9

if mo == 6:
    model_root=model_path+'Backbone_IR_152_Epoch_112_Batch_2547328_Time_2019-07-13-02-59_checkpoint.pth'
    backbone = IR_152([112, 112])
    shu=10


name_list = [name for name in os.listdir(data_dir)]


s = time.time()
print('开始')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tta = True

# load backbone from a checkpoint
print("Loading Backbone Checkpoint '{}'".format(model_root))
backbone.load_state_dict(torch.load(model_root))
backbone.to(device)

# extract features
backbone.eval()  # set to evaluation mode

face_features, cnt = extract_feature(name_list, backbone, device, tta,data_dir)
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