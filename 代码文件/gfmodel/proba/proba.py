import os
import cv2
import numpy as np
import time
from tqdm import tqdm
from PIL import Image,ImageEnhance
import os
import sys
import imp
import time
import tensorflow as tf
import math
import random
from scipy import misc

#数据路径，sub路径，存放csv路径，存放bin路径
data_dir = 'testing/Test_Data/'  # 测试集图片路径
subpath = 'testing/submission_template.csv'  # 测试集csv路径
cunpath = 'blendsub/' #存放csv文件路径
mo=1#1-2

if mo==1:
    mo= 'PFE_sphere64_msarcface_am'
    shu=11
if mo==2:
    mo='PFE_sphere64_casia_am'
    shu=12



batch_size=1
def mutual_likelihood_score_loss(labels, mu, log_sigma_sq):
    with tf.name_scope('MLS_Loss'):
        batch_size = tf.shape(mu)[0]

        diag_mask = tf.eye(batch_size, dtype=tf.bool)
        non_diag_mask = tf.logical_not(diag_mask)

        sigma_sq = tf.exp(log_sigma_sq)
        loss_mat = negative_MLS(mu, mu, sigma_sq, sigma_sq)

        label_mat = tf.equal(labels[:, None], labels[None, :])
        label_mask_pos = tf.logical_and(non_diag_mask, label_mat)

        loss_pos = tf.boolean_mask(loss_mat, label_mask_pos)

        return tf.reduce_mean(loss_pos)

class Network:
    def __init__(self):
        self.graph = tf.Graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        tf_config = tf.ConfigProto(gpu_options=gpu_options,
                                   allow_soft_placement=True, log_device_placement=False)
        self.sess = tf.Session(graph=self.graph, config=tf_config)

    def initialize(self, config, num_classes=None):
        '''
            Initialize the graph from scratch according to config.
        '''
        with self.graph.as_default():
            with self.sess.as_default():
                # Set up placeholders
                h, w = config.image_size
                channels = config.channels
                self.images = tf.placeholder(tf.float32, shape=[None, h, w, channels], name='images')
                self.labels = tf.placeholder(tf.int32, shape=[None], name='labels')

                self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
                self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
                self.phase_train = tf.placeholder(tf.bool, name='phase_train')
                self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name='global_step')

                # Initialialize the backbone network
                network = imp.load_source('embedding_network', config.embedding_network)
                mu, conv_final = network.inference(self.images, config.embedding_size)

                # Initialize the uncertainty module
                uncertainty_module = imp.load_source('uncertainty_module', config.uncertainty_module)
                log_sigma_sq = uncertainty_module.inference(conv_final, config.embedding_size,
                                                            phase_train=self.phase_train,
                                                            weight_decay=config.weight_decay,
                                                            scope='UncertaintyModule')

                self.mu = tf.identity(mu, name='mu')
                self.sigma_sq = tf.identity(tf.exp(log_sigma_sq), name='sigma_sq')

                # Build all losses
                loss_list = []
                self.watch_list = {}

                MLS_loss = mutual_likelihood_score_loss(self.labels, mu, log_sigma_sq)
                loss_list.append(MLS_loss)
                self.watch_list['loss'] = MLS_loss

                # Collect all losses
                reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='reg_loss')
                loss_list.append(reg_loss)
                self.watch_list['reg_loss'] = reg_loss

                total_loss = tf.add_n(loss_list, name='total_loss')
                grads = tf.gradients(total_loss, self.trainable_variables)

                # Training Operaters
                train_ops = []

                opt = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9)
                apply_gradient_op = opt.apply_gradients(list(zip(grads, self.trainable_variables)))

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                train_ops.extend([apply_gradient_op] + update_ops)

                train_ops.append(tf.assign_add(self.global_step, 1))
                self.train_op = tf.group(*train_ops)

                # Collect TF summary
                for k, v in self.watch_list.items():
                    tf.summary.scalar('losses/' + k, v)
                tf.summary.scalar('learning_rate', self.learning_rate)
                self.summary_op = tf.summary.merge_all()

                # Initialize variables
                self.sess.run(tf.local_variables_initializer())
                self.sess.run(tf.global_variables_initializer())
                self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=99)

        return

    @property
    def trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='UncertaintyModule')

    def save_model(self, model_dir, global_step):
        with self.sess.graph.as_default():
            checkpoint_path = os.path.join(model_dir, 'ckpt')
            metagraph_path = os.path.join(model_dir, 'graph.meta')

            print('Saving variables...')
            self.saver.save(self.sess, checkpoint_path, global_step=global_step, write_meta_graph=False)
            if not os.path.exists(metagraph_path):
                print('Saving metagraph...')
                self.saver.export_meta_graph(metagraph_path)

    def restore_model(self, model_dir, restore_scopes=None):
        var_list = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        with self.sess.graph.as_default():
            if restore_scopes is not None:
                var_list = [var for var in var_list if any([scope in var.name for scope in restore_scopes])]
            model_dir = os.path.expanduser(model_dir)
            ckpt_file = tf.train.latest_checkpoint(model_dir)

            print('Restoring {} variables from {} ...'.format(len(var_list), ckpt_file))
            saver = tf.train.Saver(var_list)
            saver.restore(self.sess, ckpt_file)

    def load_model(self, model_path, scope=None):
        with self.sess.graph.as_default():
            model_path = os.path.expanduser(model_path)

            # Load grapha and variables separatedly.
            meta_files = [file for file in os.listdir(model_path) if file.endswith('.meta')]
            assert len(meta_files) == 1
            meta_file = os.path.join(model_path, meta_files[0])
            ckpt_file = tf.train.latest_checkpoint(model_path)

            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
            saver = tf.train.import_meta_graph(meta_file, clear_devices=True, import_scope=scope)
            saver.restore(self.sess, ckpt_file)

            # Setup the I/O Tensors
            self.images = self.graph.get_tensor_by_name('images:0')
            self.phase_train = self.graph.get_tensor_by_name('phase_train:0')
            self.keep_prob = self.graph.get_tensor_by_name('keep_prob:0')
            self.mu = self.graph.get_tensor_by_name('mu:0')
            self.sigma_sq = self.graph.get_tensor_by_name('sigma_sq:0')
            self.config = imp.load_source('network_config', os.path.join(model_path, 'config.py'))

    def train(self, images_batch, labels_batch, learning_rate, keep_prob):
        feed_dict = {self.images: images_batch,
                     self.labels: labels_batch,
                     self.learning_rate: learning_rate,
                     self.keep_prob: keep_prob,
                     self.phase_train: True, }
        _, wl, sm = self.sess.run([self.train_op, self.watch_list, self.summary_op], feed_dict=feed_dict)

        step = self.sess.run(self.global_step)

        return wl, sm, step

    def extract_feature(self, images, batch_size, proc_func=None, verbose=False):
        num_images = len(images)
        num_features = self.mu.shape[1]
        mu = np.ndarray((num_images, num_features), dtype=np.float32)
        sigma_sq = np.ndarray((num_images, num_features), dtype=np.float32)
        start_time = time.time()
        for start_idx in range(0, num_images, batch_size):
            if verbose:
                elapsed_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
                sys.stdout.write('# of images: %d Current image: %d Elapsed time: %s \t\r'
                                 % (num_images, start_idx, elapsed_time))
            end_idx = min(num_images, start_idx + batch_size)
            images_batch = images[start_idx:end_idx]
            if proc_func:
                images_batch = proc_func(images_batch)
            feed_dict = {self.images: images_batch,
                         self.phase_train: False,
                         self.keep_prob: 1.0}
            mu[start_idx:end_idx], sigma_sq[start_idx:end_idx] = self.sess.run([self.mu, self.sigma_sq],
                                                                               feed_dict=feed_dict)
        if verbose:
            print('')
        return mu, sigma_sq


# Calulate the shape for creating new array given (h,w)
def get_new_shape(images, size=None, n=None):
    shape = list(images.shape)
    if size is not None:
        h, w = tuple(size)
        shape[1] = h
        shape[2] = w
    if n is not None:
        shape[0] = n
    shape = tuple(shape)
    return shape


def random_crop(images, size):
    n, _h, _w = images.shape[:3]
    h, w = tuple(size)
    shape_new = get_new_shape(images, size)
    assert (_h >= h and _w >= w)

    images_new = np.ndarray(shape_new, dtype=images.dtype)

    y = np.random.randint(low=0, high=_h - h + 1, size=(n))
    x = np.random.randint(low=0, high=_w - w + 1, size=(n))

    for i in range(n):
        images_new[i] = images[i, y[i]:y[i] + h, x[i]:x[i] + w]

    return images_new


def center_crop(images, size):
    n, _h, _w = images.shape[:3]
    h, w = tuple(size)
    assert (_h >= h and _w >= w)

    y = int(round(0.5 * (_h - h)))
    x = int(round(0.5 * (_w - w)))

    images_new = images[:, y:y + h, x:x + w]

    return images_new


def random_flip(images):
    images_new = images.copy()
    flips = np.random.rand(images_new.shape[0]) >= 0.5

    for i in range(images_new.shape[0]):
        if flips[i]:
            images_new[i] = np.fliplr(images[i])

    return images_new


def flip(images):
    images_new = images.copy()
    for i in range(images_new.shape[0]):
        images_new[i] = np.fliplr(images[i])

    return images_new


def resize(images, size):
    n, _h, _w = images.shape[:3]
    h, w = tuple(size)
    shape_new = get_new_shape(images, size)

    images_new = np.ndarray(shape_new, dtype=images.dtype)

    for i in range(n):
        images_new[i] = misc.imresize(images[i], (h, w))

    return images_new


def padding(images, padding):
    n, _h, _w = images.shape[:3]
    if len(padding) == 2:
        pad_t = pad_b = padding[0]
        pad_l = pad_r = padding[1]
    else:
        pad_t, pad_b, pad_l, pad_r = tuple(padding)

    size_new = (_h + pad_t + pad_b, _w + pad_l + pad_b)
    shape_new = get_new_shape(images, size_new)
    images_new = np.zeros(shape_new, dtype=images.dtype)
    images_new[:, pad_t:pad_t + _h, pad_l:pad_l + _w] = images

    return images_new


def standardize_images(images, standard):
    if standard == 'mean_scale':
        mean = 127.5
        std = 128.0
    elif standard == 'scale':
        mean = 0.0
        std = 255.0
    images_new = images.astype(np.float32)
    images_new = (images_new - mean) / std
    return images_new


def random_shift(images, max_ratio):
    n, _h, _w = images.shape[:3]
    pad_x = int(_w * max_ratio) + 1
    pad_y = int(_h * max_ratio) + 1
    images_temp = padding(images, (pad_x, pad_y))
    images_new = images.copy()

    shift_x = (_w * max_ratio * np.random.rand(n)).astype(np.int32)
    shift_y = (_h * max_ratio * np.random.rand(n)).astype(np.int32)

    for i in range(n):
        images_new[i] = images_temp[i, pad_y + shift_y[i]:pad_y + shift_y[i] + _h,
                        pad_x + shift_x[i]:pad_x + shift_x[i] + _w]

    return images_new


def random_downsample(images, min_ratio):
    n, _h, _w = images.shape[:3]
    images_new = images.copy()
    ratios = min_ratio + (1 - min_ratio) * np.random.rand(n)

    for i in range(n):
        w = int(round(ratios[i] * _w))
        h = int(round(ratios[i] * _h))
        images_new[i, :h, :w] = misc.imresize(images[i], (h, w))
        images_new[i] = misc.imresize(images_new[i, :h, :w], (_h, _w))

    return images_new


def random_interpolate(images):
    _n, _h, _w = images.shape[:3]
    nd = images.ndim - 1
    assert _n % 2 == 0
    n = int(_n / 2)

    ratios = np.random.rand(n, *([1] * nd))
    images_left, images_right = (images[np.arange(n) * 2], images[np.arange(n) * 2 + 1])
    images_new = ratios * images_left + (1 - ratios) * images_right
    images_new = images_new.astype(np.uint8)

    return images_new


def expand_flip(images):
    '''Flip each image in the array and insert it after the original image.'''
    _n, _h, _w = images.shape[:3]
    shape_new = get_new_shape(images, n=2 * _n)
    images_new = np.stack([images, flip(images)], axis=1)
    images_new = images_new.reshape(shape_new)
    return images_new


def five_crop(images, size):
    _n, _h, _w = images.shape[:3]
    h, w = tuple(size)
    assert h <= _h and w <= _w

    shape_new = get_new_shape(images, size, n=5 * _n)
    images_new = []
    images_new.append(images[:, :h, :w])
    images_new.append(images[:, :h, -w:])
    images_new.append(images[:, -h:, :w])
    images_new.append(images[:, -h:, -w:])
    images_new.append(center_crop(images, size))
    images_new = np.stack(images_new, axis=1).reshape(shape_new)
    return images_new


def ten_crop(images, size):
    _n, _h, _w = images.shape[:3]
    shape_new = get_new_shape(images, size, n=10 * _n)
    images_ = five_crop(images, size)
    images_flip_ = five_crop(flip(images), size)
    images_new = np.stack([images_, images_flip_], axis=1)
    images_new = images_new.reshape(shape_new)
    return images_new


register = {
    'resize': resize,
    'padding': padding,
    'random_crop': random_crop,
    'center_crop': center_crop,
    'random_flip': random_flip,
    'standardize': standardize_images,
    'random_shift': random_shift,
    'random_interpolate': random_interpolate,
    'random_downsample': random_downsample,
    'expand_flip': expand_flip,
    'five_crop': five_crop,
    'ten_crop': ten_crop,
}

from scipy import misc
def preprocess(images, config, is_training=False):
    proc_funcs = config.preprocess_train if is_training else config.preprocess_test
    for proc in proc_funcs:
        proc_name, proc_args = proc[0], proc[1:]
        images = register[proc_name](images, *proc_args)
    if len(images.shape) == 3:
        images = images[:, :, :, None]
    return images



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
    # if random_flip and np.random.choice([True, False]):
    image = np.fliplr(image)
    return image

def transf(image, network):
    images = preprocess(np.array([image]), network.config, False)
    # Run forward pass to calculate embeddings
    a, b = network.extract_feature(images, batch_size, verbose=False)
    feat = np.concatenate([a, b], axis=1)
    return feat[0]
from sklearn.preprocessing  import normalize
def get_featurs(model, test_list,data_dir):
    mp=dict()
    cnt = 0
    #     pbar = tqdm(len(test_list))
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



        jia17=  cv2.resize(image, (122, 144))[16:128, 16:128]  # 122,144
        jia17 = transf(jia17, model)

        jia18 = cv2.resize(fan, (122, 144))[16:128, 16:128]  # 122,144
        jia18 = transf(jia18, model)

        cnt += 1

        if cnt%1000==0:
            print(cnt)
        mp[img_path] = []

        image=transf(image,model)
        fan=transf(fan, model)
        jia1=transf(jia1,model)
        jia2 = transf(jia2, model)
        jia3 = transf(jia3, model)
        jia4 = transf(jia4, model)
        jia13 = transf(jia13, model)
        jia14 = transf(jia14, model)
        jia15 = transf(jia15, model)
        jia16 = transf(jia16, model)
        image=image+fan+jia1+jia2+jia3+jia4+jia13+jia14+jia15+jia16+jia17+jia18
        feature= normalize( image.reshape(1, -1), norm='l2')
        if features is None:
            features = feature
        else:
            features = np.vstack((features, feature))

    return features, cnt


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
import scipy.io as sio


modelpath = mo+'/'
# Load model files and config file
network = Network()
network.load_model(modelpath)

name_list = [name for name in os.listdir(data_dir)]

s = time.time()

face_features, cnt = get_featurs(network, name_list,data_dir)
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
sample_sub = open(subpath, 'r')
sub = open(cunpath+str(shu)+'.csv', 'w')
print('Loaded CSV')
lines = sample_sub.readlines()
for line in lines:
    pair = line.split(',')[0]
    sub.write(pair + ',')
    a, b = pair.split(':')
    score = '%f' % cosin_metric(face_features[a][0], face_features[b][0])
    sub.write(score + '\n')
sample_sub.close()