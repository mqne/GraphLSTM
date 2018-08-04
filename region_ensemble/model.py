# coding: utf-8

# Based on `0914-ren-mid-aug-deeper1`
# 
# Trained with new training data

# # Navigate
# 
# - [PCA](#PCA)
# - [Model](#Model)
# - [Train](#Train)
# - [Plotly](#Plotly)
# - [Load Weights](#Load-Weights)
# - [Validate](#Validate)
# - [Explore Validate](#Explore-Validate)
# - [Explore Test](#Explore-Test)
# - [Test All](#Test-All)
# - [Generate Zip](#Generate-Zip)

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns

# get_ipython().run_line_magic('matplotlib', 'inline')
#sns.set_style("whitegrid")

# In[2]:


import plotly.offline as py
import plotly.graph_objs as go

#py.init_notebook_mode(connected=True)

# In[3]:


# get_ipython().run_line_magic('env', 'CUDA_VISIBLE_DEVICES=1') todo: adapt to non ipython

import tensorflow as tf
from keras.utils import Progbar
from keras.models import Model, Sequential, load_model
from keras.backend import set_session
import h5py
import scipy.misc
import math
import random
import pandas as pd
from os import path, makedirs
import itertools
from sklearn.decomposition import PCA as SKLEARN_PCA
import numpy as np
import scipy as sp

from tqdm import tqdm
from sys import stdout


# In[4]:


class Const:
    DEVICE = '/gpu:0'
    SRC_IMAGE_SHAPE = (250, 250, 1)
    MODEL_IMAGE_SHAPE = (128, 128, 1)
    LABEL_SHAPE = (21 * 3)

    NUM_TRAIN_SAMPLES = 956587
    NUM_VALIDATE_SAMPLES = 0
    NUM_TEST_SAMPLES = 295510
    TRAIN_BATCH_SIZE = 256
    VALIDATE_BATCH_SIZE = 256
    TEST_BATCH_SIZE = 256
    NUM_TRAIN_BATCHES = math.ceil(NUM_TRAIN_SAMPLES / TRAIN_BATCH_SIZE)
    NUM_VALIDATE_BATCHES = math.ceil(NUM_VALIDATE_SAMPLES / VALIDATE_BATCH_SIZE)
    NUM_TEST_BATCHES = math.ceil(NUM_TEST_SAMPLES / TEST_BATCH_SIZE)

    WEIGHT_DECAY = 0.000005
    NUM_EIGENVECTORS = 40

    AUGMENT_ROTATE_RANGE = (-np.pi / 2, np.pi / 2)  # uniform distribution
    AUGMENT_TRANSLATE_MEAN_SD = (0, 10)  # normal distribution
    AUGMENT_SCALE_MEAN_SD = (1, 0.05)  # normal distribution


# In[5]:


# prefix = "01-retest"

# dataset_root = r"/mnt/nasbi/shared/research/hand-pose-estimation/hands2017/data/hand2017_nor_img_new"
# train_list = ["nor_%08d.pkl" % i for i in range(1000, 957001, 1000)] + ["nor_00957032.pkl"]
# validate_list = []

# testset_root = r"/mnt/nasbi/shared/research/hand-pose-estimation/hands2017/data/hand2017_test_0914"
# test_list = ["%08d.pkl" % i for i in range(10000, 290001, 10000)] + ["00295510.pkl"]


# In[6]:


def resize_image(image, to_shape):
    image = image.reshape([image.shape[0], image.shape[1]])  # convert to 2D
    image = scipy.misc.imresize(image, size=(to_shape[0], to_shape[1]), interp='nearest')  # resize 2D
    image = image.reshape(to_shape)  # convert to final shape (3D)
    return image


def sample_generator(dataset_root, container_dir, container_name_list, resize_to_shape=None, progress_desc=None,
                     leave=False):
    SHAPE_DICT = {
        'image': Const.SRC_IMAGE_SHAPE,
        'pose': Const.LABEL_SHAPE,
    }
    DTYPE_DICT = {
        'image': float,
        'pose': float,
    }

    #     p = Progbar(len(container_name_list))  # DEBUG: ProgressBar
    container_list = map(lambda container_name: path.join(dataset_root, container_dir, container_name),
                         container_name_list)
    it = enumerate(container_list)
    if progress_desc is not None:
        it = tqdm(it, total=len(container_name_list), desc=progress_desc, leave=leave, dynamic_ncols=True)
    for i, container in it:
        #         p.update(i)  # DEBUG: ProgressBar
        sample_seq = pd.read_pickle(container, compression='gzip')
        for sample in sample_seq:
            sample = sample.reshape(SHAPE_DICT[container_dir]).astype(DTYPE_DICT[container_dir])
            # Resize
            if resize_to_shape is not None:
                sample = resize_image(sample, resize_to_shape)
            yield sample


# test_params = pd.read_pickle(path.join(testset_root, "tran_para_img.pkl"), compression='gzip')

def param_and_name_generator(dataset_root, container_dir, container_name_list):
    container_list = map(lambda container_name: path.join(dataset_root, container_dir, container_name),
                         container_name_list)
    for i, container in enumerate(container_list):
        sample_seq = pd.read_pickle(container, compression='gzip')
        for sample, name in zip(sample_seq, sample_seq.index):
            yield sample, name


# In[7]:


import region_ensemble.transformations as tfms


def get_augment_params():
    rot = np.random.uniform(*Const.AUGMENT_ROTATE_RANGE)
    trans = np.random.normal(*Const.AUGMENT_TRANSLATE_MEAN_SD, 2).tolist()
    scale = np.random.normal(*Const.AUGMENT_SCALE_MEAN_SD)
    return rot, trans, scale


def alter_pair(image, points):
    center = [image.shape[1] / 2, image.shape[0] / 2]
    rot, trans, scale = get_augment_params()
    image_alt, points_alt = tfms.transform_image_and_points(np.asarray(image).reshape([image.shape[0], image.shape[1]]),
                                                            np.asarray(points).reshape([-1, 3]),
                                                            center=center,
                                                            rot=rot,
                                                            trans=trans,
                                                            scale=scale)
    return image_alt.reshape(image.shape), points_alt.reshape(points.shape)


def alter_label(points, center):
    rot, trans, scale = get_augment_params()
    image_alt, points_alt = tfms.transform_image_and_points(None,
                                                            np.asarray(points).reshape([-1, 3]),
                                                            center=center,
                                                            rot=rot,
                                                            trans=trans,
                                                            scale=scale)
    return points_alt.reshape(points.shape)


def unzip_pair(zipped):
    """
    input:  ((a1, b1), (a2, b2), (a3, b3), ...)
    output: (a1, a2, a3, ...), (b1, b2, b3, ...)
    """
    seq_gen = zip(*zipped)
    return next(seq_gen), next(seq_gen)


def pair_batch_generator_one_epoch(dataset_root, container_name_list, batch_size, shuffle=False, augmented=False,
                                   progress_desc=None, leave=False, epoch=-1):
    if shuffle:
        container_name_list = np.random.permutation(container_name_list)
    image_generator = sample_generator(dataset_root, "image", container_name_list,
                                       progress_desc=progress_desc, leave=leave)
    label_generator = sample_generator(dataset_root, "pose", container_name_list)
    # batch loop
    while True:
        # pickup
        image_islice = itertools.islice(image_generator, batch_size)
        label_islice = itertools.islice(label_generator, batch_size)
        # generate
        image_list = list(image_islice)
        label_list = list(label_islice)
        if len(image_list) == 0:
            # end of epoch
            break

        # process
        if augmented and epoch != 0:
            image_list, label_list = unzip_pair(map(alter_pair, image_list, label_list))
        image_list = [resize_image(image, Const.MODEL_IMAGE_SHAPE) for image in image_list]

        yield np.asarray(image_list), np.asarray(label_list)


def pair_batch_generator(dataset_root, container_name_list, batch_size, shuffle=False, augmented=False,
                         progress_desc=None, leave=False):
    # epoch loop
    for i_epoch in range(65535):
        pair_batch_generator_one_epoch(dataset_root=dataset_root, container_name_list=container_name_list,
                                       batch_size=batch_size, shuffle=shuffle, augmented=augmented,
                                       progress_desc=progress_desc, leave=leave, epoch=i_epoch)


def image_batch_generator(dataset_root, container_name_list, batch_size, progress_desc=None, leave=False):
    # while True:
        image_batch_generator_one_epoch(dataset_root=dataset_root, container_name_list=container_name_list,
                                        batch_size=batch_size, progress_desc=progress_desc, leave=leave)


def image_batch_generator_one_epoch(dataset_root, container_name_list, batch_size, progress_desc=None, leave=False):
    image_generator = sample_generator(dataset_root, "image", container_name_list,
                                       progress_desc=progress_desc, leave=leave)
    # batch loop
    while True:
        # pickup
        image_islice = itertools.islice(image_generator, batch_size)
        # generate
        image_list = list(image_islice)

        if len(image_list) == 0:
            # end of epoch
            break

        image_list = [resize_image(image, Const.MODEL_IMAGE_SHAPE) for image in image_list]
        yield np.asarray(image_list)


# # PCA

class RegEnPCA:
    """Helper class for doing PCA for the Region Ensemble network.

     Original implementation by Kai Akiyama, Robotics Vision Lab, NAIST.
     """

    @staticmethod
    def get_mean_and_eigenvectors(train_labels):
        OUTPUT_DIM = Const.LABEL_SHAPE
        pca = SKLEARN_PCA(n_components=OUTPUT_DIM)
        print("Fitting PCA …", end=" ")
        stdout.flush()
        pca.fit(train_labels)
        print("done.")
        pca_mean = pca.mean_
        # Get eigenvectors as vertical vectors
        pca_eigenvalues, pca_eigenvectors_original = np.linalg.eig(pca.get_covariance())
        # Transpose to get eigenvectors as horizontal vectors
        pca_eigenvectors = pca_eigenvectors_original.transpose()
        # Sort by eignvalues
        eigenpairs = sorted(zip(pca_eigenvalues, pca_eigenvectors))[::-1]
        pca_eigenvalues, pca_eigenvectors = zip(*eigenpairs)
        return pca_mean, np.asarray(pca_eigenvectors), np.asarray(pca_eigenvalues)

    @staticmethod
    def get_mean_eigenvectors_eigenvalues_with_augment(label_sample_gen, augment_times=0):
        """
        augment_times=0 ... no augmentation used in RegEnPCA
        """
        total_num_labels = Const.NUM_TRAIN_SAMPLES * (1 + augment_times)  # hard coded train_list_size, beware
        pca_label_array = np.zeros([total_num_labels, Const.LABEL_SHAPE])
        center = [Const.SRC_IMAGE_SHAPE[1] / 2, Const.SRC_IMAGE_SHAPE[0] / 2]
        i = 0
        for label in label_sample_gen:
            pca_label_array[i] = label
            i += 1
            for _ in range(augment_times):
                pca_label_array[i] = alter_label(label, center)
                i += 1
        assert (i == total_num_labels)
        return RegEnPCA.get_mean_and_eigenvectors(pca_label_array)

    def __init__(self, directory_prefix, use_precalculated_samples=False, dataset_root=None, train_list=None):
        self._directory_prefix = directory_prefix
        if not use_precalculated_samples:
            if dataset_root is None:
                raise ValueError("Must define `dataset_root` directory when reading samples for PCA calculation")
            if train_list is None:
                raise ValueError("Must define train_list namespace when reading samples for PCA calculation")
            train_label_gen = sample_generator(dataset_root, "pose", train_list, progress_desc="Generating PCA samples")
            self._pca_mean, self._pca_eigenvectors, self._pca_eigenvalues = \
                self.get_mean_eigenvectors_eigenvalues_with_augment(train_label_gen, augment_times=2)
            self._tofile()
        else:
            self._fromfile()

    def _tofile(self):
        if not path.exists(self._directory_prefix):
            makedirs(self._directory_prefix)
            print("Created new directory `%s` to store PCA data." % self._directory_prefix)
        self._pca_mean.tofile("%s/pca_mean.npy" % self._directory_prefix)
        self._pca_eigenvectors.tofile("%s/pca_eigenvectors.npy" % self._directory_prefix)
        self._pca_eigenvalues.tofile("%s/pca_eigenvalues.npy" % self._directory_prefix)

    def _fromfile(self):
        self._pca_mean = np.fromfile("%s/pca_mean.npy" % self._directory_prefix)
        self._pca_eigenvectors = np.fromfile("%s/pca_eigenvectors.npy" % self._directory_prefix).reshape(
            [Const.LABEL_SHAPE, Const.LABEL_SHAPE])
        self._pca_eigenvalues = np.fromfile("%s/pca_eigenvalues.npy" % self._directory_prefix)

    def plot(self):
        plt.subplot(2, 1, 1)
        plt.bar(np.arange(len(self._pca_eigenvalues)), self._pca_eigenvalues)
        plt.subplot(2, 1, 2)
        plt.bar(np.arange(len(self._pca_eigenvalues)), self._pca_eigenvalues)
        plt.yscale('log')

    @property
    def mean(self):
        return self._pca_mean

    @property
    def eigenvectors(self):
        return self._pca_eigenvectors

    @property
    def eigenvalues(self):
        return self._pca_eigenvalues


# # Model

class RegEnModel(Model):
    """Region Ensemble network model.

     Original implementation by Kai Akiyama, Robotics Vision Lab, NAIST.
     """

    def __init__(self, directory_prefix, result_layer=True):
        super().__init__(*self._build_model(result_layer=result_layer))
        self._directory_prefix = directory_prefix

    @property
    def directory_prefix(self):
        return self._directory_prefix

    @staticmethod
    def _build_model(result_layer=True):
        from keras.layers import Input, Convolution2D, MaxPooling2D, add, Lambda, Dense, Flatten, Dropout, concatenate
        from keras import regularizers

        with tf.device(Const.DEVICE):
            image = Input(shape=Const.MODEL_IMAGE_SHAPE, name="regen_net_input")

            com = Convolution2D(filters=32,
                                kernel_size=(5, 5),
                                padding='same',
                                activation='relu',
                                )(image)

            com = Convolution2D(filters=32,
                                kernel_size=(3, 3),
                                padding='same',
                                activation='relu',
                                )(com)  # todo: this was changed from (image) to (com), as model is probably better that way, but: comparability to Kai's evaluated model?

            com = MaxPooling2D(pool_size=(2, 2),
                               padding='same'
                               )(com)

            shortcut = com  # 👇

            com = Convolution2D(filters=64,
                                kernel_size=(3, 3),
                                padding='same',
                                activation='relu',
                                kernel_regularizer=regularizers.l2(Const.WEIGHT_DECAY),
                                )(com)

            com = Convolution2D(filters=64,
                                kernel_size=(3, 3),
                                padding='same',
                                activation='relu',
                                kernel_regularizer=regularizers.l2(Const.WEIGHT_DECAY),
                                )(com)

            shortcut = Convolution2D(filters=int(com.shape[-1]),  # 👈
                                     kernel_size=(1, 1),
                                     padding='same',
                                     kernel_regularizer=regularizers.l2(Const.WEIGHT_DECAY),
                                     )(shortcut)
            com = add([com, shortcut])

            com = MaxPooling2D(pool_size=(2, 2),
                               padding='same'
                               )(com)

            shortcut = com  # 👇

            com = Convolution2D(filters=128,
                                kernel_size=(3, 3),
                                padding='same',
                                activation='relu',
                                kernel_regularizer=regularizers.l2(Const.WEIGHT_DECAY),
                                )(com)

            com = Convolution2D(filters=128,
                                kernel_size=(3, 3),
                                padding='same',
                                activation='relu',
                                kernel_regularizer=regularizers.l2(Const.WEIGHT_DECAY),
                                )(com)

            shortcut = Convolution2D(filters=int(com.shape[-1]),  # 👈
                                     kernel_size=(1, 1),
                                     padding='same',
                                     kernel_regularizer=regularizers.l2(Const.WEIGHT_DECAY),
                                     )(shortcut)
            com = add([com, shortcut])

            com = MaxPooling2D(pool_size=(2, 2),
                               padding='same'
                               )(com)

            shortcut = com  # 👇

            com = Convolution2D(filters=256,
                                kernel_size=(3, 3),
                                padding='same',
                                activation='relu',
                                kernel_regularizer=regularizers.l2(Const.WEIGHT_DECAY),
                                )(com)

            com = Convolution2D(filters=256,
                                kernel_size=(3, 3),
                                padding='same',
                                activation='relu',
                                kernel_regularizer=regularizers.l2(Const.WEIGHT_DECAY),
                                )(com)

            shortcut = Convolution2D(filters=int(com.shape[-1]),  # 👈
                                     kernel_size=(1, 1),
                                     padding='same',
                                     kernel_regularizer=regularizers.l2(Const.WEIGHT_DECAY),
                                     )(shortcut)
            com = add([com, shortcut])

            com = MaxPooling2D(pool_size=(2, 2),
                               padding='same'
                               )(com)

            # Here we got 8x8x256 features

            def segment_features(feat):
                """
                Split features into 4 blocks for both direction (total 16 blocks),
                and then concatenate any adjacent 2x2 blocks to form one segment.
                Therefore feat.shape[1] and [2] must be divisible by 4.
                """
                assert (len(feat.shape) == 4)

                rows = int(feat.shape[1])
                cols = int(feat.shape[2])
                assert (rows % 4 == 0)
                assert (cols % 4 == 0)

                block_rows = int(rows / 4)
                block_cols = int(cols / 4)

                segments = []
                for i_row in range(0, block_rows * 3, block_rows):
                    for i_col in range(0, block_cols * 3, block_cols):
                        segments += [feat[:, i_row:i_row + block_rows * 2, i_col:i_col + block_cols * 2, :]]

                return segments

            segments = Lambda(segment_features)(com)

            segments = [Convolution2D(filters=256,
                                      kernel_size=(3, 3),
                                      padding='same',
                                      activation='relu',
                                      kernel_regularizer=regularizers.l2(Const.WEIGHT_DECAY),
                                      )(seg)
                        for seg in segments]

            segments = [Flatten()(seg) for seg in segments]

            segments = [Dense(units=2048,
                              activation='relu',
                              kernel_regularizer=regularizers.l2(Const.WEIGHT_DECAY),
                              )(seg)
                        for seg in segments]
            segments = [Dropout(rate=0.5)
                        (seg)
                        for seg in segments]

            segments = [Dense(units=2048,
                              activation='relu',
                              kernel_regularizer=regularizers.l2(Const.WEIGHT_DECAY),
                              )(seg)
                        for seg in segments]
            segments = [Dropout(rate=0.5)
                        (seg)
                        for seg in segments]

            # Concatenating segments

            com = concatenate(segments, axis=-1)

            # Dense

            com = Dense(units=2048,
                        kernel_regularizer=regularizers.l2(Const.WEIGHT_DECAY)
                        )(com)
            com = Dropout(rate=0.5)(com)

            # PCA bottleneck

            com = Dense(units=Const.NUM_EIGENVECTORS,
                        kernel_regularizer=regularizers.l2(Const.WEIGHT_DECAY)
                        )(com)

            # Result

            if result_layer:
                com = Dense(units=Const.LABEL_SHAPE,
                            kernel_regularizer=regularizers.l2(Const.WEIGHT_DECAY),
                            )(com)

            predict = com

        return image, predict

    # Set PCA bottleneck weights
    def set_pca_bottleneck_weights(self, reg_en_pca):
        w = self.get_weights()
        w[-1] = reg_en_pca.mean
        w[-2] = reg_en_pca.eigenvectors[:Const.NUM_EIGENVECTORS]
        self.set_weights(w)


# Smooth L1 loss function from Fan's implementation.
def soft_loss(y_true, y_pred):
    x = tf.abs(y_true - y_pred)
    x = tf.cast(x, tf.float32)
    x_bool = tf.cast(tf.less_equal(x, 1.), tf.float32)
    loss = tf.reduce_mean(x_bool * (0.5 * x * x) + (1 - x_bool) * 1. * (x - 0.5))
    return loss


# # Train

def lr_schedule(epoch):
    initial = 0.005
    multiply = 0.1
    after_every = 20.
    sustain = math.pow(multiply, 1. / after_every)
    lr = initial * (sustain ** epoch)
    print("lr=%f" % lr)
    return lr


# # Plotly

def nodes_to_finger_edges(nodes):
    nodes = nodes.reshape([21, 3])

    bone_structure = np.asarray([[0, 1, 6, 7, 8],
                                 [0, 2, 9, 10, 11],
                                 [0, 3, 12, 13, 14],
                                 [0, 4, 15, 16, 17],
                                 [0, 5, 18, 19, 20], ])
    nodes_x = nodes[..., 0]
    nodes_y = nodes[..., 1]
    nodes_z = nodes[..., 2]
    edges_x = nodes_x[np.asarray(bone_structure)]
    edges_y = nodes_y[np.asarray(bone_structure)]
    edges_z = nodes_z[np.asarray(bone_structure)]
    edges_x = np.hstack([edges_x, [[None]] * 5]).flatten()
    edges_y = np.hstack([edges_y, [[None]] * 5]).flatten()
    edges_z = np.hstack([edges_z, [[None]] * 5]).flatten()
    return edges_x, edges_y, edges_z


def plot_scatter2d(image, pred=None, true=None):
    image = image.reshape(Const.SRC_IMAGE_SHAPE[0], Const.SRC_IMAGE_SHAPE[1])
    pred = pred.reshape([21, 3]) if pred is not None else None
    true = true.reshape([21, 3]) if true is not None else None

    data = []

    image_heatmap = go.Heatmap(
        name="image",
        z=image,
        showscale=False,
        # colorscale='Greys',
    )
    data += [image_heatmap]

    if pred is not None:
        pred_edges = nodes_to_finger_edges(pred)
        pred_scatter = go.Scatter(
            name="predicted",
            x=pred_edges[0], y=pred_edges[1],
            mode='lines+markers',
            marker={'size': 10, 'opacity': 0.75},
        )
        data += [pred_scatter]
    if true is not None:
        true_edges = nodes_to_finger_edges(true)
        true_scatter = go.Scatter(
            name="true",
            x=true_edges[0], y=true_edges[1],
            mode='lines+markers',
            marker={'size': 10, 'opacity': 0.75},
        )
        data += [true_scatter]
    layout = go.Layout(
        width=700,
        height=500,
        yaxis={'autorange': 'reversed'},
        margin={'l': 0, 'r': 200, 'b': 0, 't': 0},
    )
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig)


def plot_scatter3d(image, pred=None, true=None):
    image = image.reshape(Const.SRC_IMAGE_SHAPE[0], Const.SRC_IMAGE_SHAPE[1])
    pred = pred.reshape([21, 3]) if pred is not None else None
    true = true.reshape([21, 3]) if true is not None else None

    x, y = np.meshgrid(np.arange(0, image.shape[1]), np.arange(0, image.shape[0]))
    x = x[0 < image]
    y = y[0 < image]
    z = image[0 < image]

    data = []

    image_scatter = go.Scatter3d(
        name="image",
        x=x, y=y, z=z,
        mode='markers',
        marker={'size': 2, 'opacity': 0.2}
    )
    data += [image_scatter]

    if pred is not None:
        pred_edges = nodes_to_finger_edges(pred)
        pred_scatter = go.Scatter3d(
            name="predicted",
            x=pred_edges[0], y=pred_edges[1], z=pred_edges[2],
            mode='lines+markers',
            marker={'size': 4, 'opacity': 0.75}
        )
        data += [pred_scatter]
    if true is not None:
        true_edges = nodes_to_finger_edges(true)
        true_scatter = go.Scatter3d(
            name="true",
            x=true_edges[0], y=true_edges[1], z=true_edges[2],
            mode='lines+markers',
            marker={'size': 4, 'opacity': 0.75}
        )
        data += [true_scatter]
    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
    )
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig)


# # Load Weights

# In[12]:


#region_ensemble_net.load_weights('./%s/model.70.hdf5' % prefix)  # , custom_objects={'soft_loss': soft_loss})

# # Validate

# In[13]:


#validate_image_batch_gen = image_batch_generator(dataset_root, validate_list, Const.VALIDATE_BATCH_SIZE)

# In[ ]:


#predictions = region_ensemble_net.predict_generator(
#    validate_image_batch_gen,
#    steps=Const.NUM_VALIDATE_BATCHES
#)

# In[ ]:


# mean absolute error
#validate_label_gen = sample_generator(dataset_root, "pose", validate_list)
#validate_label = np.asarray(list(validate_label_gen))
#print("average", np.abs(validate_label - predictions).mean())

# # Explore Validate

# In[16]:


#validate_model_image_gen = sample_generator(dataset_root, "image", train_list, resize_to_shape=Const.MODEL_IMAGE_SHAPE)
#validate_src_image_gen = sample_generator(dataset_root, "image", train_list)
#validate_label_gen = sample_generator(dataset_root, "pose", train_list)

# In[17]:


#validate_model_image = next(validate_model_image_gen)
#validate_src_image = next(validate_src_image_gen)
#validate_label = next(validate_label_gen)
#validate_uvd = region_ensemble_net.predict(np.asarray([validate_model_image]))

# In[18]:


#plot_scatter3d(validate_src_image, pred=validate_uvd, true=validate_label)

# # Explore Test

# In[17]:


#test_model_image_gen = sample_generator(testset_root, "image", test_list, resize_to_shape=Const.MODEL_IMAGE_SHAPE)
#test_src_image_gen = sample_generator(testset_root, "image", test_list)

# In[18]:


#test_model_image = next(test_model_image_gen)
#test_src_image = next(test_src_image_gen)
#test_uvd = region_ensemble_net.predict(np.asarray([test_model_image]))
#plot_scatter2d(test_src_image, pred=test_uvd)
#plot_scatter3d(test_src_image, pred=test_uvd)


# # Test All

# In[14]:


def uvd2xyz(uvd):
    fx = 475.065948
    fy = 475.065857
    cx = 315.944855
    cy = 245.287079

    xyz = np.empty_like(uvd)
    for i in range(uvd.shape[0]):
        u = uvd[i, 0]
        v = uvd[i, 1]
        d = uvd[i, 2]
        xyz[i, 0] = (u - cx) * d / fx
        xyz[i, 1] = (v - cy) * d / fy
        xyz[i, 2] = d
    return xyz


def transform(pose_coor, paras):
    pose_coor = np.array(pose_coor).reshape([21, 3])  # estimation coordinates
    img_min_coor = paras[0]  # 2 elements  # paras are read from tran_para_img.pkl
    coor_tran = paras[1]  # 2 elements
    img_min = paras[2]  # scalar
    window1 = paras[3]  # 4 elements
    scale = paras[4]  # scalar

    pose_coor[:, [0, 1]] = pose_coor[:, [1, 0]]
    tran_v = pose_coor[:, :2] - coor_tran
    tran_v = tran_v / scale

    p_true = np.zeros([21, 3])
    p_true[:, :2] = img_min_coor + tran_v  # transfer back to box coordinates
    p_true[:, :2] += np.array([window1[1], window1[0]])  # transfer back to original picture coordinates

    p_true[:, [0, 1]] = p_true[:, [1, 0]]
    p_true[:, 2] = img_min + pose_coor[:, 2]

    xyz = uvd2xyz(p_true)
    xyz = xyz.reshape(63)

    return xyz


# In[15]:


# # Generate Zip

# In[16]:


def test_xyz_and_name_gen(model, testset_root, test_list):

    test_image_batch_gen = image_batch_generator(testset_root, test_list, Const.TEST_BATCH_SIZE)
    # test_params is already loaded

    test_uvd = model.predict_generator(
        test_image_batch_gen,
        steps=Const.NUM_TEST_BATCHES,
        max_queue_size=1000,
        use_multiprocessing=True,
        verbose=True,
    )

    test_param_and_name_gen = param_and_name_generator(testset_root, 'tran_para_img', test_list)

    for uvd, param_and_name in zip(test_uvd, test_param_and_name_gen):
        parm = param_and_name[0]
        name = param_and_name[1]
        xyz = transform(uvd, parm)
        yield xyz, name


#pose_submit_gen = ("frame\\images\\{}\t{}".format(name, '\t'.join(map(str, xyz)))
#                   for xyz, name
#                   in test_xyz_and_name_gen())
#
#np.savetxt('./%s/result-newtest.txt' % prefix, np.asarray(list(pose_submit_gen)), delimiter='\n', fmt="%s")

import zipfile

#with zipfile.ZipFile("./%s/result-newtest.zip" % prefix, 'w', zipfile.ZIP_DEFLATED) as zf:
#    zf.write("./%s/result-newtest.txt" % prefix, "result.txt")
