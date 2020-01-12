from __future__ import division
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
import os
import multiprocessing
workers = multiprocessing.cpu_count()-1

_IS_TF_2 = True
from tensorflow import keras
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LambdaCallback
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.client import device_lib
from tensorflow.keras.regularizers import l2

from collections import Counter

from sklearn.utils import class_weight
import cv2
import glob
import random
from tqdm import tqdm
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels
import itertools
import pickle

from model import Deeplabv3

def icnr_weights(init = tf.keras.initializers.GlorotNormal(), scale=2, shape=[3,3,32,4], dtype = tf.float32):
    sess = tf.Session()
    return sess.run(ICNR(init, scale=scale)(shape=shape, dtype=dtype))

class ICNR:
    """ICNR initializer for checkerboard artifact free sub pixel convolution
    Ref:
     [1] Andrew Aitken et al. Checkerboard artifact free sub-pixel convolution
     https://arxiv.org/pdf/1707.02937.pdf)
    Args:
    initializer: initializer used for sub kernels (orthogonal, glorot uniform, etc.)
    scale: scale factor of sub pixel convolution
    """

    def __init__(self, initializer, scale=1):
        self.scale = scale
        self.initializer = initializer

    def __call__(self, shape, dtype, partition_info=None):
        shape = list(shape)
        if self.scale == 1:
            return self.initializer(shape)

        new_shape = shape[:3] + [shape[3] // (self.scale ** 2)]
        x = self.initializer(new_shape, dtype, partition_info)
        x = tf.transpose(x, perm=[2, 0, 1, 3])
        x = tf.image.resize_nearest_neighbor(x, size=(shape[0] * self.scale, shape[1] * self.scale))
        x = tf.space_to_depth(x, block_size=self.scale)
        x = tf.transpose(x, perm=[1, 2, 0, 3])

        return x

class Subpixel(Conv2D):
    def __init__(self,
                 filters,
                 kernel_size,
                 r,
                 padding='valid',
                 data_format=None,
                 strides=(1,1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Subpixel, self).__init__(
            filters=r*r*filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.r = r

    def _phase_shift(self, I):
        r = self.r
        bsize, a, b, c = I.get_shape().as_list()
        bsize = K.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
        X = K.reshape(I, [bsize, a, b, int(c/(r*r)),r, r]) # bsize, a, b, c/(r*r), r, r
        X = K.permute_dimensions(X, (0, 1, 2, 5, 4, 3))  # bsize, a, b, r, r, c/(r*r)
        #Keras backend does not support tf.split, so in future versions this could be nicer
        X = [X[:,i,:,:,:,:] for i in range(a)] # a, [bsize, b, r, r, c/(r*r)
        X = K.concatenate(X, 2)  # bsize, b, a*r, r, c/(r*r)
        X = [X[:,i,:,:,:] for i in range(b)] # b, [bsize, r, r, c/(r*r)
        X = K.concatenate(X, 2)  # bsize, a*r, b*r, c/(r*r)
        return X

    def call(self, inputs):
        return self._phase_shift(super(Subpixel, self).call(inputs))

    def compute_output_shape(self, input_shape):
        unshifted = super(Subpixel, self).compute_output_shape(input_shape)
        return (unshifted[0], self.r*unshifted[1], self.r*unshifted[2], int(unshifted[3]/(self.r*self.r)))

    def get_config(self):
        config = super(Conv2D, self).get_config()
        config.pop('rank')
        config.pop('dilation_rate')
        config['filters']= int(config['filters'] / self.r*self.r)
        config['r'] = self.r
        return config

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    trained_classes = classes
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize=11)
    tick_marks = np.arange(len(classes))
    plt.xticks(np.arange(len(trained_classes)), classes, rotation=90,fontsize=9)
    plt.yticks(tick_marks, classes,fontsize=9)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, np.round(cm[i, j],2), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=7)
    plt.tight_layout()
    plt.ylabel('True label',fontsize=9)
    plt.xlabel('Predicted label',fontsize=9)
    return cm

# Fully connected CRF post processing function
def do_crf(im, mask, zero_unsure=True):
    colors, labels = np.unique(mask, return_inverse=True)
    image_size = mask.shape[:2]
    n_labels = len(set(labels.flat))
    d = dcrf.DenseCRF2D(image_size[1], image_size[0], n_labels)  # width, height, nlabels
    U = unary_from_labels(labels, n_labels, gt_prob=.7, zero_unsure=zero_unsure)
    d.setUnaryEnergy(U)
    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3,3), compat=3)
    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    # im is an image-array, e.g. im.dtype == np.uint8 and im.shape == (640,480,3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=im.astype('uint8'), compat=10)
    Q = d.inference(5) # 5 - num of iterations
    MAP = np.argmax(Q, axis=0).reshape(image_size)
    unique_map = np.unique(MAP)
    for u in unique_map: # get original labels back
        np.putmask(MAP, MAP == u, colors[u])
    return MAP
    # MAP = do_crf(frame, labels.astype('int32'), zero_unsure=False)
    
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def sparse_crossentropy_ignoring_last_label(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    y_true = K.one_hot(tf.cast(y_true[:,:,0], tf.int32), nb_classes+1)[:,:,:-1]
    return K.categorical_crossentropy(y_true, y_pred)

def sparse_accuracy_ignoring_last_label(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))
    y_true = tf.cast(K.flatten(y_true), tf.int64)
    legal_labels = ~K.equal(y_true, nb_classes)
    return K.sum(tf.cast(legal_labels & K.equal(y_true, 
                                                    K.argmax(y_pred, axis=-1)), tf.float32)) / K.sum(tf.cast(legal_labels, tf.float32))
def Jaccard(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    iou = []
    pred_pixels = K.argmax(y_pred, axis=-1)
    for i in range(0, nb_classes): # exclude first label (background) and last label (void)
        true_labels = K.equal(y_true[:,:,0], i)
        pred_labels = K.equal(pred_pixels, i)
        inter = tf.cast(true_labels & pred_labels, tf.int32)
        union = tf.cast(true_labels | pred_labels, tf.int32)
        legal_batches = K.sum(tf.cast(true_labels, tf.int32), axis=1)>0
        ious = K.sum(inter, axis=1)/K.sum(union, axis=1)
        if _IS_TF_2:
            iou.append(K.mean(ious[legal_batches]))
        else:
            iou.append(K.mean(tf.gather(ious, indices=tf.where(legal_batches)))) # returns average IoU of the same objects
    iou = tf.stack(iou)
    legal_labels = ~tf.math.is_nan(iou) if _IS_TF_2 else ~tf.debugging.is_nan(iou)
    iou = iou[legal_labels] if _IS_TF_2 else tf.gather(iou, indices=tf.where(legal_labels))
    return K.mean(iou)

class SegModel:
    epochs = 20
    batch_size = 16
    def __init__(self, dataset='VOCdevkit/VOC2012', image_size=(256,256)):
        self.sz = image_size
        self.mainpath = dataset
        self.crop = False
            
    
    def create_seg_model(self, net, n=183, backbone='mobilenetv2', load_weights=False, multi_gpu=True):
        
        '''
        Net is:
        1. original deeplab v3+
        2. original deeplab v3+ and subpixel upsampling layer
        '''
        model = Deeplabv3(weights='pascal_voc', input_shape=self.sz + (3,), classes=n,
                        backbone=backbone, OS=16)
        
        base_model = Model(model.input, model.layers[-5].output)
        self.net = net
        self.modelpath = '{}_{}.h5'.format(backbone, net)
        if backbone=='xception':
            scale = 4
        else:
            scale = 8
        if net == 'original':
            x = Conv2D(n, (1, 1), padding='same', name='conv_upsample')(base_model.output)
            x = Lambda(lambda x: tf.compat.v1.image.resize_bilinear(x,size=(self.sz[0],self.sz[1])))(x)
            x = Reshape((self.sz[0]*self.sz[1], n)) (x)
            x = Activation('softmax', name = 'pred_mask')(x)
            model = Model(base_model.input, x, name='deeplabv3p')
        elif net == 'subpixel':
            x = Subpixel(n, 1, scale, padding='same')(base_model.output)
            x = Reshape((self.sz[0]*self.sz[1], n)) (x)
            x = Activation('softmax', name = 'pred_mask')(x)
            model = Model(base_model.input, x, name='deeplabv3p_subpixel')
        # Do ICNR
        for layer in model.layers:
            if type(layer) == Subpixel:
                c, b = layer.get_weights()
                w = icnr_weights(scale=scale, shape=c.shape)
                layer.set_weights([w, b])
                
        if load_weights:
            model.load_weights('{}_{}.h5'.format(backbone, net))

        if multi_gpu:
            print("Multi GPU Mode activated")
            # from tensorflow.keras.utils import multi_gpu_model
            # model = multi_gpu_model(model, gpus = 2)
            
        self.model = model
        return model

    # def create_generators(self, crop_shape=False, mode='train', do_ahisteq=True, n_classes=21, horizontal_flip=True, 
    #                       vertical_flip=False, blur=False, with_bg=True, brightness=0.1, rotation=5.0, 
    #                       zoom=0.1, validation_split=.2, seed=7):
                
    #     generator = SegmentationGenerator(folder = self.mainpath, mode = mode, n_classes = n_classes, do_ahisteq = do_ahisteq,
    #                                    batch_size=self.batch_size, resize_shape=self.sz[::-1], crop_shape=crop_shape, 
    #                                    horizontal_flip=horizontal_flip, vertical_flip=vertical_flip, blur = blur,
    #                                    brightness=brightness, rotation=rotation, zoom=zoom,
    #                                    validation_split = validation_split, seed = seed)
                
    #     return generator

    def load_weights(self, model):
        model.load_weights(self.modelpath)
        
    def train_generator(self, model, train_generator, valid_generator, callbacks, mp = True):
        steps = len(train_generator)
        h = model.fit_generator(train_generator,
                                steps_per_epoch=steps, 
                                epochs = self.epochs, verbose=1, 
                                callbacks = callbacks, 
                                validation_data=valid_generator, 
                                validation_steps=len(valid_generator), 
                                max_queue_size=10, 
                                workers=workers, use_multiprocessing=mp)
        return h
    
    def train(self, model, X, y, val_data, tf_board = False, plot_train_process = True):
        h = model.fit(X, y, validation_data = val_data, verbose=1, 
                      batch_size = self.batch_size, epochs = self.epochs, 
                      callbacks = self.build_callbacks(tf_board = tf_board, plot_process = plot_train_process))
        return h
    
    @classmethod
    def set_num_epochs(cls, new_epochs):
        cls.epochs = new_epochs
    @classmethod
    def set_batch_size(cls, new_batch_size):
        cls.batch_size = new_batch_size

def image_batch_generator(images, gt_images, batch_size, mode='train'):
    while True:
        X = np.zeros((batch_size, image_size[1], image_size[0], 3), dtype='float32')
        SW = np.zeros((batch_size, image_size[1]*image_size[0]), dtype='float32')
        Y = np.zeros((batch_size, image_size[1]*image_size[0], 1), dtype='float32')
        F = np.zeros((batch_size, image_size[1]*image_size[0], 1), dtype='float32')
        F_SW = np.zeros((batch_size, image_size[1]*image_size[0]), dtype='float32')

        batch_paths = np.random.choice(a=len(images), size=batch_size)
        for n, i in enumerate(batch_paths):
            image = images[i]
            label = gt_images[i]
            labels = np.unique(label)

            if mode == 'train':
                horizontal_flip=True
                blur = 0
                vertical_flip=0
                brightness=0.1
                rotation=5.0
                zoom=0.1
                do_ahisteq = True

                if blur and random.randint(0,1):
                    image = cv2.GaussianBlur(image, (blur, blur), 0)
                    
                # Do augmentation
                if horizontal_flip and random.randint(0,1):
                    image = cv2.flip(image, 1)
                    label = cv2.flip(label, 1)
                if vertical_flip and random.randint(0,1):
                    image = cv2.flip(image, 0)
                    label = cv2.flip(label, 0)
                if brightness:
                    factor = 1.0 + random.gauss(mu=0.0, sigma=brightness)
                    if random.randint(0,1):
                        factor = 1.0/factor
                    table = np.array([((i / 255.0) ** factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
                    image = cv2.LUT(image, table)
                if rotation:
                    angle = random.gauss(mu=0.0, sigma=rotation)
                else:
                    angle = 0.0
                if zoom:
                    scale = random.gauss(mu=1.0, sigma=zoom)
                else:
                    scale = 1.0
                if rotation or zoom:
                    M = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), angle, scale)
                    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
                    label = cv2.warpAffine(label, M, (label.shape[1], label.shape[0]))

                if do_ahisteq: # and convert to RGB
                    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
                    image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR) # to BGR
                        
            label = label.astype('int32')
            for j in np.setxor1d(np.unique(label), labels):
                label[label==j] = n_classes
            
            y = label.flatten()
            y[y>(n_classes-1)]=n_classes    
            Y[n]  = np.expand_dims(y, -1)
            F[n]  = (Y[n]!=0).astype('float32') # get all pixels that aren't background
            valid_pixels = F[n][Y[n]!=n_classes] # get all pixels (bg and foregroud) that aren't void
            u_classes = np.unique(valid_pixels)
            class_weights = class_weight.compute_class_weight('balanced', u_classes, valid_pixels)
            class_weights = {class_id : w for class_id, w in zip(u_classes, class_weights)}
            if len(class_weights)==1: # no bg\no fg
                if 1 in u_classes:
                    class_weights[0] = 0.
                else:
                    class_weights[1] = 0.
            elif not len(class_weights):
                class_weights[0] = 0.
                class_weights[1] = 0.

            sw_valid = np.ones(y.shape)
            np.putmask(sw_valid, Y[n]==0, class_weights[0]) # background weights
            np.putmask(sw_valid, F[n], class_weights[1]) # foreground wegihts 
            np.putmask(sw_valid, Y[n]==n_classes, 0)
            F_SW[n] = sw_valid
            X[n] = image    

            # Create adaptive pixels weights
            filt_y = y[y!=n_classes]
            u_classes = np.unique(filt_y)
            if len(u_classes):
                class_weights = class_weight.compute_class_weight('balanced', u_classes, filt_y)
                class_weights = {class_id : w for class_id, w in zip(u_classes, class_weights)}
            class_weights[n_classes] = 0.
            for yy in u_classes:
                np.putmask(SW[n], y==yy, class_weights[yy])
                
            np.putmask(SW[n], y==n_classes, 0)   

        sample_dict = {'pred_mask' : SW}
        yield (X, Y, sample_dict)

# class SegmentationGenerator(Sequence):
    
#     def __init__(self, folder='/vinai/hoanganh/coco2017/', mode='train', n_classes=183, batch_size=16, resize_shape=None, 
#                  validation_split = .1, seed = 7, crop_shape=None, horizontal_flip=True, blur = 0,
#                  vertical_flip=0, brightness=0.1, rotation=5.0, zoom=0.1, do_ahisteq = True):
        
#         self.blur = blur
#         self.histeq = do_ahisteq
#         # self.image_path_list = sorted(glob.glob(os.path.join(folder, 'JPEGImages', 'train', '*')))
#         # self.label_path_list = sorted(glob.glob(os.path.join(folder, 'SegmentationClassAug', '*')))

#         # np.random.seed(seed)
        
#         # n_images_to_select = round(len(self.image_path_list) * validation_split)
#         # x = np.random.permutation(len(self.image_path_list))[:n_images_to_select]
#         # if mode == 'train':
#         #     x = np.setxor1d(x, np.arange(len(self.image_path_list)))
            
#         # self.image_path_list = [self.image_path_list[j] for j in x]
#         # self.label_path_list = [self.label_path_list[j] for j in x]
        
#         # if mode == 'test':
#         #     self.image_path_list = sorted(glob.glob(os.path.join(folder, 'JPEGImages', 'test', '*')))[:100]
#         if mode=='train':
#             self.image_list, self.label_list = map(list, zip(*pickle.load(open(os.path.join(folder, 'train_images_1_255.pkl'), 'rb'))))
#         else:
#             self.image_list, self.label_list = map(list, zip(*pickle.load(open(os.path.join(folder, 'val_images_255.pkl'), 'rb'))))
#         self.mode = mode
#         self.n_classes = n_classes
#         self.batch_size = batch_size
#         self.resize_shape = resize_shape
#         self.crop_shape = crop_shape
#         self.horizontal_flip = horizontal_flip
#         self.vertical_flip = vertical_flip
#         self.brightness = brightness
#         self.rotation = rotation
#         self.zoom = zoom
#         self.part = 0
#         self.num_training_part = 8
#         self.folder = folder
#         # Preallocate memory
#         if self.crop_shape:
#             self.X = np.zeros((batch_size, crop_shape[1], crop_shape[0], 3), dtype='float32')
#             self.SW = np.zeros((batch_size, crop_shape[1]*crop_shape[0]), dtype='float32')
#             self.Y = np.zeros((batch_size, crop_shape[1]*crop_shape[0], 1), dtype='float32')
#             self.F = np.zeros((batch_size, crop_shape[1]*crop_shape[0], 1), dtype='float32')
#             self.F_SW = np.zeros((batch_size, crop_shape[1]*crop_shape[0]), dtype='float32')
#         elif self.resize_shape:
#             self.X = np.zeros((batch_size, resize_shape[1], resize_shape[0], 3), dtype='float32')
#             self.SW = np.zeros((batch_size, resize_shape[1]*resize_shape[0]), dtype='float32')
#             self.Y = np.zeros((batch_size, resize_shape[1]*resize_shape[0], 1), dtype='float32')
#             self.F = np.zeros((batch_size, resize_shape[1]*resize_shape[0], 1), dtype='float32')
#             self.F_SW = np.zeros((batch_size, resize_shape[1]*resize_shape[0]), dtype='float32')
#         else:
#             raise Exception('No image dimensions specified!')
        
#     def __len__(self):
#         # return len(self.image_path_list) // self.batch_size
#         return len(self.image_list) // self.batch_size
        
#     def __getitem__(self, i):
        
#         # for n, (image_path, label_path) in enumerate(zip(self.image_path_list[i*self.batch_size:(i+1)*self.batch_size], 
#         #                                                 self.label_path_list[i*self.batch_size:(i+1)*self.batch_size])):
            
#         #     image = cv2.imread(image_path, 1)
#         #     label = cv2.imread(label_path, 0)
#         for n, (image, label) in enumerate(zip(self.image_list[i*self.batch_size:(i+1)*self.batch_size], 
#                                                         self.label_list[i*self.batch_size:(i+1)*self.batch_size])):
#             labels = np.unique(label)
            
#             if self.blur and random.randint(0,1):
#                 image = cv2.GaussianBlur(image, (self.blur, self.blur), 0)

#             if self.resize_shape and not self.crop_shape:
#                 image = cv2.resize(image, self.resize_shape)
#                 label = cv2.resize(label, self.resize_shape, interpolation = cv2.INTER_NEAREST)
        
#             if self.crop_shape:
#                 image, label = _random_crop(image, label, self.crop_shape)
                
#             # Do augmentation
#             if self.horizontal_flip and random.randint(0,1):
#                 image = cv2.flip(image, 1)
#                 label = cv2.flip(label, 1)
#             if self.vertical_flip and random.randint(0,1):
#                 image = cv2.flip(image, 0)
#                 label = cv2.flip(label, 0)
#             if self.brightness:
#                 factor = 1.0 + random.gauss(mu=0.0, sigma=self.brightness)
#                 if random.randint(0,1):
#                     factor = 1.0/factor
#                 table = np.array([((i / 255.0) ** factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
#                 image = cv2.LUT(image, table)
#             if self.rotation:
#                 angle = random.gauss(mu=0.0, sigma=self.rotation)
#             else:
#                 angle = 0.0
#             if self.zoom:
#                 scale = random.gauss(mu=1.0, sigma=self.zoom)
#             else:
#                 scale = 1.0
#             if self.rotation or self.zoom:
#                 M = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), angle, scale)
#                 image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
#                 label = cv2.warpAffine(label, M, (label.shape[1], label.shape[0]))

#             if self.histeq: # and convert to RGB
#                 img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
#                 img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
#                 image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR) # to BGR
                 
#             label = label.astype('int32')
#             for j in np.setxor1d(np.unique(label), labels):
#                 label[label==j] = self.n_classes
            
#             y = label.flatten()
#             y[y>(self.n_classes-1)]=self.n_classes
                            
#             self.Y[n]  = np.expand_dims(y, -1)
#             self.F[n]  = (self.Y[n]!=0).astype('float32') # get all pixels that aren't background
#             valid_pixels = self.F[n][self.Y[n]!=self.n_classes] # get all pixels (bg and foregroud) that aren't void
#             u_classes = np.unique(valid_pixels)
#             class_weights = class_weight.compute_class_weight('balanced', u_classes, valid_pixels)
#             class_weights = {class_id : w for class_id, w in zip(u_classes, class_weights)}
#             if len(class_weights)==1: # no bg\no fg
#                 if 1 in u_classes:
#                     class_weights[0] = 0.
#                 else:
#                     class_weights[1] = 0.
#             elif not len(class_weights):
#                 class_weights[0] = 0.
#                 class_weights[1] = 0.
        
#             sw_valid = np.ones(y.shape)
#             np.putmask(sw_valid, self.Y[n]==0, class_weights[0]) # background weights
#             np.putmask(sw_valid, self.F[n], class_weights[1]) # foreground wegihts 
#             np.putmask(sw_valid, self.Y[n]==self.n_classes, 0)
#             self.F_SW[n] = sw_valid
#             self.X[n] = image    
        
#             # Create adaptive pixels weights
#             filt_y = y[y!=self.n_classes]
#             u_classes = np.unique(filt_y)
#             if len(u_classes):
#                 class_weights = class_weight.compute_class_weight('balanced', u_classes, filt_y)
#                 class_weights = {class_id : w for class_id, w in zip(u_classes, class_weights)}
#             class_weights[self.n_classes] = 0.
#             for yy in u_classes:
#                 np.putmask(self.SW[n], y==yy, class_weights[yy])
                
#             np.putmask(self.SW[n], y==self.n_classes, 0)

#         sample_dict = {'pred_mask' : self.SW}
#         return self.X, self.Y, sample_dict
        
#     def on_epoch_end(self):
#         self.part += 1
#         if self.mode =='train':
#             del self.image_list, self.label_list
#             print("\nTRAINING ON PART " + str(self.part%self.num_training_part + 1))
#             self.image_list, self.label_list = map(list, zip(*pickle.load(open(os.path.join(self.folder, 'train_images_' + str(self.part%self.num_training_part + 1) + '_255.pkl'), 'rb'))))
#         else:
#             # c = list(zip(self.image_path_list, self.label_path_list))
#             # c = list(zip(self.image_list, self.label_list))
#             # random.shuffle(c)
#             # self.image_path_list, self.label_path_list = zip(*c)
#             # self.image_list, self.label_list = zip(*c)
#             print("\n NOT IN TRAINING MODE")
    
def _random_crop(image, label, crop_shape):
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        raise Exception('Image and label must have the same dimensions!')
        
    if (crop_shape[0] < image.shape[1]) and (crop_shape[1] < image.shape[0]):
        x = random.randrange(image.shape[1]-crop_shape[0])
        y = random.randrange(image.shape[0]-crop_shape[1])
        
        return image[y:y+crop_shape[1], x:x+crop_shape[0], :], label[y:y+crop_shape[1], x:x+crop_shape[0]]
    else:
        image = cv2.resize(image, crop_shape)
        label = cv2.resize(label, crop_shape, interpolation = cv2.INTER_NEAREST)
        return image, label

def build_callbacks(tf_board = False):
    tensorboard = TensorBoard(log_dir='./logs/'+SegClass.net, histogram_freq=0,
                        write_graph=False, write_images = False)
    checkpointer = ModelCheckpoint(filepath = SegClass.modelpath, verbose=1, save_best_only=True, save_weights_only=True,
                                    monitor = 'val_{}'.format(monitor), mode = mode)
    stop_train = EarlyStopping(monitor = 'val_{}'.format(monitor), patience=100, verbose=1, mode = mode)
    reduce_lr = ReduceLROnPlateau(monitor = 'val_{}'.format(monitor), factor=0.5,
                patience=5, min_lr=1e-6)
    if tf_board:
        callbacks = [reduce_lr, stop_train, tensorboard]
    else:
        callbacks = [checkpointer, reduce_lr, stop_train]
    return callbacks

def load_data(i):
    image_list, label_list = map(list, zip(*pickle.load(open(os.path.join(PATH, 'train_images_' + str(i) + '_255.pkl'), 'rb'))))
    return image_list, label_list

if __name__ == '__main__':

    image_size = (256, 256) #(512,512) (720, 1280)
    batch_size = 10

    better_model = False
    load_pretrained_weights = False

    losses = sparse_crossentropy_ignoring_last_label
    metrics = {'pred_mask' : [Jaccard, sparse_accuracy_ignoring_last_label]}

    backbone = 'mobilenetv2' #mobilenetv2, xception

    NET = 'deeplab_' + backbone
    PATH = '/vinai/hoanganh/coco2017/'
    NUM_TRAINING_PART = 8
    n_classes = 183

    print('Num workers:', workers)
    print('Backbone:', backbone)
    print('Path to dataset:', PATH)
    print('N classes:', n_classes)
    print('Image size:', image_size)
    print('Batch size:', batch_size)

    config = tf.compat.v1.ConfigProto(gpu_options = 
                        tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
        # device_count = {'GPU': 1}
        )
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)

    SegClass = SegModel(PATH, image_size)
    SegClass.set_batch_size(batch_size)
    
    if better_model:
        model = SegClass.create_seg_model(net='subpixel', n=n_classes, \
                                        multi_gpu=True, backbone=backbone)
    else:
        model = SegClass.create_seg_model(net='original', n=n_classes,\
                                        multi_gpu=True, backbone=backbone)

    # model.load_weights(SegClass.modelpath)
    # model.summary()
    model.compile(optimizer = Adam(lr=1e-3, epsilon=1e-8, decay=1e-6), sample_weight_mode = "temporal",
                loss = losses, metrics = metrics)
    print('Weights path:', SegClass.modelpath)

    # train_generator = SegClass.create_generators(blur=5,crop_shape=None, mode='train_1', n_classes=n_classes,
    #                                              horizontal_flip=True, vertical_flip=False, brightness=0.3, 
    #                                              rotation=False, zoom=0.1, validation_split=.15, seed=7, do_ahisteq=False)

    # valid_generator = SegClass.create_generators(blur=0, crop_shape=None, mode='val', 
    #                                             n_classes=n_classes, horizontal_flip=True, vertical_flip=False, 
    #                                             brightness=.1, rotation=False, zoom=.05, validation_split=.15, 
    #                                             seed=7, do_ahisteq=False)

    monitor = 'Jaccard'
    mode = 'max'

    # fine-tune model (train only last conv layers)
    if load_pretrained_weights:
        flag = 0
        for k, l in enumerate(model.layers):
            l.trainable = False
            if l.name == 'concat_projection':
                flag = 1
            if flag:
                l.trainable = True
            


    callbacks = build_callbacks(tf_board = False)
    # for i in range(100):
    #     print("Loading training part " + str(i%NUM_TRAINING_PART + 1))
    #     train_generator = SegClass.create_generators(blur=5,crop_shape=None, mode='train_' + str(i%NUM_TRAINING_PART + 1), n_classes=n_classes,
    #                                                 horizontal_flip=True, vertical_flip=False, brightness=0.3, 
    #                                                 rotation=False, zoom=0.1, validation_split=.15, seed=7, do_ahisteq=False)
    #     SegClass.set_num_epochs(2)
    #     history = SegClass.train_generator(model, train_generator, valid_generator, callbacks, mp = True)
    # train_generator = SegClass.create_generators(blur=5,crop_shape=None, mode='train', n_classes=n_classes,
    #                                             horizontal_flip=True, vertical_flip=False, brightness=0.3, 
    #                                             rotation=False, zoom=0.1, validation_split=.15, seed=7, do_ahisteq=False)
    # SegClass.set_num_epochs(1000)
    # history = SegClass.train_generator(model, train_generator, valid_generator, callbacks, mp = False)
    images_val, gt_images_val = map(list, zip(*pickle.load(open(os.path.join(PATH, 'val_images_255.pkl'), 'rb'))))
    
    for i in range(100):
        print("\nLoading training part " + str(i%NUM_TRAINING_PART + 1))
        images, gt_images = load_data(i%NUM_TRAINING_PART + 1)
        model.fit_generator(generator=image_batch_generator(images, gt_images, batch_size, mode='train'),
                        steps_per_epoch=len(images)//batch_size,
                        epochs=2,
                        verbose=1,
                        validation_data=image_batch_generator(images_val, gt_images_val, batch_size, mode='val'),
                        validation_steps=len(images_val)//batch_size,
                        callbacks=callbacks)
        del images, gt_images