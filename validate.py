import graph_lstm as glstm
import region_ensemble.model as re
from helpers import *

import tensorflow as tf
import keras.backend as K
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, TensorBoard

import numpy as np
import seaborn as sns
import zipfile

from tqdm import tqdm
import os


# set plot style
# sns.set_style("whitegrid")


# dataset path declarations

prefix = "train-02"
checkpoint_dir = r"/data2/GraphLSTM/%s" % prefix

dataset_root = r"/data2/datasets/hands2017/data/hand2017_nor_img_new"
train_list = ["nor_%08d.pkl" % i for i in range(1000, 957001, 1000)] + ["nor_00957032.pkl"]
validate_list = []

testset_root = r"/data2/datasets/hands2017/data/hand2017_test_0914"
test_list = ["%08d.pkl" % i for i in range(10000, 290001, 10000)] + ["00295510.pkl"]

# number of timesteps to be simulated (each step, the same data is fed)
graphlstm_timesteps = 2

model_name = "regen41_graphlstm1t%i" % graphlstm_timesteps


# # PREPARE SESSION

config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


# # LOAD MODEL

print("\n###   Loading Model: %s   ###\n" % model_name)

epoch = 2

input_shape = [None, *re.Const.MODEL_IMAGE_SHAPE]
output_shape = [None, len(HAND_GRAPH_HANDS2017_INDEX_DICT), GLSTM_NUM_UNITS]


# # RUN VALIDATION

with sess.as_default():

    print("Loading meta graph …")
    loader = tf.train.import_meta_graph(checkpoint_dir + "/%s.meta" % model_name)

    print("Restoring weights for epoch %i …" % epoch)
    loader.restore(sess, checkpoint_dir + "/%s-%i" % (model_name, epoch))

    print("Getting necessary tensors …")
    input_tensor, output_tensor, groundtruth_tensor, train_step, loss = tf.get_collection(COLLECTION)

    print("Initializing variables …")
    sess.run(tf.global_variables_initializer())

    validate_list = [train_list[i] for i in range(0, len(train_list), 1000)]  # todo BAD CODE replace this
    validate_image_batch_gen = re.image_batch_generator_one_epoch(dataset_root,
                                                                  validate_list,
                                                                  re.Const.VALIDATE_BATCH_SIZE,
                                                                  progress_desc="Collecting network output",
                                                                  leave=True)
    predictions = None
    for batch in validate_image_batch_gen:
        X = batch
        actual_batch_size = X.shape[0]
        X = X.reshape([actual_batch_size, *input_shape[1:]])
        # Y = Y.reshape([actual_batch_size, *output_shape[1:]])

        batch_predictions = sess.run(output_tensor, feed_dict={input_tensor: X, K.learning_phase(): 0})
        if predictions is not None:
            predictions = np.concatenate((predictions, batch_predictions))
        else:
            predictions = batch_predictions

        # todo: pass K.learning_phase(): 1 to feed_dict (for testing: 0)


# # CALCULATE RESULTS

# mean absolute error

# get ground truth labels
validate_label_gen = re.sample_generator(dataset_root, "pose", validate_list)
validate_label = np.asarray(list(validate_label_gen))
# reshape from [set_size, 63] to [set_size, 21, 3]
validate_label = np.reshape(validate_label, [-1, *output_shape[-2:]])

print("Mean prediction error:", np.abs(validate_label - predictions).mean())  # todo which unit is this in?

print("Validation done.")
exit(0)





# # Validate

validate_image_batch_gen = re.image_batch_generator(dataset_root, validate_list, re.Const.VALIDATE_BATCH_SIZE)

predictions = region_ensemble_net.predict_generator(
    validate_image_batch_gen,
    steps=re.Const.NUM_VALIDATE_BATCHES, verbose=1
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
validate_uvd = region_ensemble_net.predict(np.asarray([validate_model_image]))

re.plot_scatter3d(validate_src_image, pred=validate_uvd, true=validate_label)