import graph_lstm as glstm
import region_ensemble.model as re
import multiple_hypotheses_extension as mhp
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

prefix, model_name, epoch = get_prefix_model_name_optionally_epoch()

# dataset path declarations

checkpoint_dir = r"/home/matthias-k/GraphLSTM_data/%s" % prefix

dataset_root = r"/home/matthias-k/datasets/hands2017/data/hand2017_nor_img_new"
train_and_validate_list = ["nor_%08d.pkl" % i for i in range(1000, 957001, 1000)] + ["nor_00957032.pkl"]

train_list, validate_list = train_validate_split(train_and_validate_list)

testset_root = r"/data2/datasets/hands2017/data/hand2017_test_0914"
test_list = ["%08d.pkl" % i for i in range(10000, 290001, 10000)] + ["00295510.pkl"]

# number of timesteps to be simulated (each step, the same data is fed)
graphlstm_timesteps = 2
learning_rate = 1e-3

checkpoint_dir += r"/%s" % model_name
tensorboard_dir = checkpoint_dir + r"/tensorboard/validation"


# # PREPARE SESSION

config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


# # LOAD MODEL

print("\n###   Loading Model: %s   ###\n" % model_name)
epoch_str = "Last epoch" if epoch is None else ("Epoch %i" % epoch)
print("##   %s   ##\n" % epoch_str)

# None loads last epoch, int loads specific epoch
# epoch = None

input_shape = [None, *re.Const.MODEL_IMAGE_SHAPE]
output_shape = [None, len(HAND_GRAPH_HANDS2017_INDEX_DICT), GLSTM_NUM_UNITS]

# gather tensorboard tensors
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)
    print("Created new tensorboard validation directory `%s`." % tensorboard_dir)
validation_summary_writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)


# # RUN VALIDATION

with sess.as_default():

    print("Loading meta graph …")
    loader = tf.train.import_meta_graph(checkpoint_dir + "/%s.meta" % model_name)

    if epoch is None:
        print("Restoring weights for last epoch …")
        loader.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
    else:
        print("Restoring weights for epoch %i …" % epoch)
        loader.restore(sess, checkpoint_dir + "/%s-%i" % (model_name, epoch))

    print("Getting necessary tensors …")
    input_tensor, output_tensor, groundtruth_tensor, train_step, loss, merged, is_training = tf.get_collection(COLLECTION)

    validate_image_batch_gen = re.image_batch_generator_one_epoch(dataset_root,
                                                                  validate_list,
                                                                  re.Const.VALIDATE_BATCH_SIZE,
                                                                  progress_desc="Collecting network output",
                                                                  leave=True)
    predictions = None
    global_step = 0
    for batch in validate_image_batch_gen:
        X = batch
        actual_batch_size = X.shape[0]
        X = X.reshape([actual_batch_size, *input_shape[1:]])
        # Y = Y.reshape([actual_batch_size, *output_shape[1:]])
        Y_dummy = np.zeros([actual_batch_size, 21, 3])  # necessary as the restored "merged" tensor computes the loss

        batch_predictions, summary = sess.run([output_tensor, merged], feed_dict={input_tensor: X,
                                                                                  groundtruth_tensor: Y_dummy,
                                                                                  K.learning_phase(): 0,
                                                                                  is_training: False})
        if predictions is not None:
            predictions = np.concatenate((predictions, batch_predictions))
        else:
            predictions = batch_predictions
        validation_summary_writer.add_summary(summary, global_step=global_step)
        global_step += 1

# # STORE PREDICTION RESULTS

npyname = predictions_npy_name(model_name, epoch)
print("Storing prediction results at %s …" % npyname)

np.save(tensorboard_dir + "/" + npyname, predictions)


# # CALCULATE RESULTS

# mean absolute error

print("Calculating errors …")

# get ground truth labels
validate_label_gen = re.sample_generator(dataset_root, "pose", validate_list)
validate_label = np.asarray(list(validate_label_gen))
# reshape from [set_size, 63] to [set_size, 21, 3]
validate_label = np.reshape(validate_label, [-1, *output_shape[-2:]])

predictions_mean, _ = np_mean_and_variance(predictions)

# each individual error [ validate_set_length, 21, 3 ]
individual_error = np.abs(validate_label - predictions_mean)

# overall error
overall_mean_error = ErrorCalculator.overall_mean_error(individual_error)

# np.save(tensorboard_dir + "/individual_error_%s%s.npy" % (model_name,
#                                                           (("_epoch" + str(epoch)) if epoch is not None else "")),
#         individual_error)

print("\n# %s" % epoch_str)
print("Mean mean prediction error (euclidean):", overall_mean_error)  # todo which unit is this in?

pred_joint_avg = np.mean(predictions_mean, axis=0)
actual_joint_avg = np.mean(validate_label, axis=0)

print("Actual joints position average:\n%r" % actual_joint_avg)
print("Predicted joints position average:\n%r" % pred_joint_avg)

print("Validation done.")
print("Point tensorboard to %s to get more insights. This directory also holds the .npy files for error analysis."
      % tensorboard_dir)
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