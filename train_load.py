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

prefix, model_name, load_epoch = get_prefix_model_name_and_epoch()

# dataset path declarations

# prefix = "train-02"
checkpoint_dir = r"/home/matthias-k/GraphLSTM_data/%s" % prefix

dataset_root = r"/home/matthias-k/datasets/hands2017/data/hand2017_nor_img_new"
train_and_validate_list = ["nor_%08d.pkl" % i for i in range(1000, 957001, 1000)] + ["nor_00957032.pkl"]

train_list, validate_list = train_validate_split(train_and_validate_list)

testset_root = r"/data2/datasets/hands2017/data/hand2017_test_0914"
test_list = ["%08d.pkl" % i for i in range(10000, 290001, 10000)] + ["00295510.pkl"]

# number of timesteps to be simulated (each step, the same data is fed)
graphlstm_timesteps = 2
learning_rate = 1e-3

#model_name = "regen41_graphlstm1t%i_outputscaling21x3wb_adamlr%f" % (graphlstm_timesteps, learning_rate)
# model_name = "regen41_graphlstm1t%i_fcrelu4d4d1_adamlr%f" % (graphlstm_timesteps, learning_rate)

checkpoint_dir += r"/%s" % model_name
tensorboard_dir = checkpoint_dir + r"/tensorboard"


# # PREPARE SESSION

config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


# # LOAD MODEL

print("\n###   Loading Model: %s   ###\n" % model_name)

max_epoch = 100
# load_epoch = 2

input_shape = [None, *re.Const.MODEL_IMAGE_SHAPE]
output_shape = [None, len(HAND_GRAPH_HANDS2017_INDEX_DICT), GLSTM_NUM_UNITS]


# # TRAIN

t = TQDMHelper()

with sess.as_default():
    print("Loading meta graph …")
    loader = tf.train.import_meta_graph(checkpoint_dir + "/%s.meta" % model_name)
    print("Restoring weights for epoch %i …" % load_epoch)
    loader.restore(sess, checkpoint_dir + "/%s-%i" % (model_name, load_epoch))
    print("Getting necessary tensors …")
    input_tensor, output_tensor, groundtruth_tensor, train_step, loss, merged = tf.get_collection(COLLECTION)
    print("Creating variable saver …")
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=2, filename=checkpoint_dir)
    print("Creating training summary writer …")
    training_summary_writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)
    print("Resuming training.")

    # only valid for default train_validate_split of 0.8
    samples_per_epoch_split80 = 765848
    batches_per_epoch_split80 = 2992
    global_step = batches_per_epoch_split80 * load_epoch

    for epoch in range(load_epoch + 1, max_epoch):
        t.start()
        training_sample_generator = re.pair_batch_generator_one_epoch(dataset_root, train_list,
                                                                      re.Const.TRAIN_BATCH_SIZE,
                                                                      shuffle=True, progress_desc="Epoch %i" % epoch,
                                                                      leave=False,
                                                                      epoch=epoch - 1)  # todo augmented=True?

        for batch in training_sample_generator:
            X, Y = batch
            actual_batch_size = X.shape[0]
            X = X.reshape([actual_batch_size, *input_shape[1:]])
            Y = Y.reshape([actual_batch_size, *output_shape[1:]])

            _, loss_value, summary = sess.run([train_step, loss, merged], feed_dict={input_tensor: X,
                                                                                     groundtruth_tensor: Y,
                                                                                     K.learning_phase(): 1})

            training_summary_writer.add_summary(summary, global_step=global_step)
            global_step += 1
            t.write("Current loss: %f" % loss_value)

            # todo: pass K.learning_phase(): 1 to feed_dict (for testing: 0)
        t.stop()
        print("Training loss after epoch %i: %f" % (epoch, loss_value))
        if epoch % 2 == 0:
            saver.save(sess, save_path=checkpoint_dir + "/%s" % model_name, global_step=epoch)

print("Training done, exiting.")
exit(0)
