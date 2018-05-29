# this file verifies that the modified model.py file works as expected

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

import sys
sys.path.append("..")

import region_ensemble.model as re

import tensorflow as tf
from keras.backend import set_session
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, TensorBoard

import numpy as np
import seaborn as sns
import zipfile


# set plot style
sns.set_style("whitegrid")


# get_ipython().run_line_magic('env', 'CUDA_VISIBLE_DEVICES=1') todo: adapt to non ipython


# dataset path declarations

prefix = "01-retest"

dataset_root = r"/mnt/nasbi/shared/research/hand-pose-estimation/hands2017/data/hand2017_nor_img_new"
train_list = ["nor_%08d.pkl" % i for i in range(1000, 957001, 1000)] + ["nor_00957032.pkl"]
validate_list = []

testset_root = r"/mnt/nasbi/shared/research/hand-pose-estimation/hands2017/data/hand2017_test_0914"
test_list = ["%08d.pkl" % i for i in range(10000, 290001, 10000)] + ["00295510.pkl"]


# # Model

# set Keras session
config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# instantiate model

pca = re.RegEnPCA(read_samples=True)
model = re.RegEnModel()

model.compile(optimizer=Adam(), loss=re.soft_loss)

# todo: possible before model.compile? If yes: include in _build_model? If no: override model.compile?
model.set_pca_bottleneck_weights(pca)

# store PNG image of model
plot_model(model, to_file='%s/model.png' % prefix, show_shapes=True)


# # Train

train_batch_gen = re.pair_batch_generator(dataset_root, train_list, re.Const.TRAIN_BATCH_SIZE, shuffle=True, augmented=True)
validate_batch_gen = re.pair_batch_generator(dataset_root, validate_list, re.Const.VALIDATE_BATCH_SIZE)


history = model.fit_generator(
    train_batch_gen,
    steps_per_epoch=re.Const.NUM_TRAIN_BATCHES,
    epochs=200,
    initial_epoch=0,
    callbacks=[
        #         LearningRateScheduler(lr_schedule),
        TensorBoard(log_dir="./%s" % prefix),
        ModelCheckpoint(
            filepath='./%s/model.{epoch:02d}.hdf5' % prefix,
        ),
    ]
)

"""
# # Load Weights

model.load_weights('./%s/model.70.hdf5' % prefix)  # , custom_objects={'soft_loss': soft_loss})


# # Validate

validate_image_batch_gen = re.image_batch_generator(dataset_root, validate_list, re.Const.VALIDATE_BATCH_SIZE)

predictions = model.predict_generator(
    validate_image_batch_gen,
    steps=re.Const.NUM_VALIDATE_BATCHES
)

# mean absolute error
validate_label_gen = re.sample_generator(dataset_root, "pose", validate_list)
validate_label = np.asarray(list(validate_label_gen))
print("average", np.abs(validate_label - predictions).mean())


# # Explore Validate

validate_model_image_gen = re.sample_generator(dataset_root, "image", train_list, resize_to_shape=re.Const.MODEL_IMAGE_SHAPE)
validate_src_image_gen = re.sample_generator(dataset_root, "image", train_list)
validate_label_gen = re.sample_generator(dataset_root, "pose", train_list)

validate_model_image = next(validate_model_image_gen)
validate_src_image = next(validate_src_image_gen)
validate_label = next(validate_label_gen)
validate_uvd = model.predict(np.asarray([validate_model_image]))

re.plot_scatter3d(validate_src_image, pred=validate_uvd, true=validate_label)


# # Explore Test

test_model_image_gen = re.sample_generator(testset_root, "image", test_list, resize_to_shape=re.Const.MODEL_IMAGE_SHAPE)
test_src_image_gen = re.sample_generator(testset_root, "image", test_list)

test_model_image = next(test_model_image_gen)
test_src_image = next(test_src_image_gen)
test_uvd = model.predict(np.asarray([test_model_image]))
re.plot_scatter2d(test_src_image, pred=test_uvd)
re.plot_scatter3d(test_src_image, pred=test_uvd)


# # Test All

test_image_batch_gen = re.image_batch_generator(testset_root, test_list, re.Const.TEST_BATCH_SIZE)
# test_params is already loaded

test_uvd = model.predict_generator(
    test_image_batch_gen,
    steps=re.Const.NUM_TEST_BATCHES,
    max_queue_size=1000,
    use_multiprocessing=True,
    verbose=True,
)


# # Generate Zip

test_param_and_name_gen = re.param_and_name_generator(testset_root, 'tran_para_img', test_list)


pose_submit_gen = ("frame\\images\\{}\t{}".format(name, '\t'.join(map(str, xyz)))
                   for xyz, name
                   in re.test_xyz_and_name_gen())

np.savetxt('./%s/result-newtest.txt' % prefix, np.asarray(list(pose_submit_gen)), delimiter='\n', fmt="%s")

with zipfile.ZipFile("./%s/result-newtest.zip" % prefix, 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write("./%s/result-newtest.txt" % prefix, "result.txt")
"""
