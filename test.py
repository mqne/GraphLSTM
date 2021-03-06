# run a trained network on the HIM2017 test set and collect predictions in a .npy file

import region_ensemble.model as re
from helpers import *
import dataset_loaders

import tensorflow as tf
import keras.backend as K

import numpy as np

import os


prefix, model_name, epoch = get_prefix_model_name_optionally_epoch()


checkpoint_dir = r"/home/matthias-k/GraphLSTM_data/%s" % prefix

# number of timesteps to be simulated (each step, the same data is fed)
graphlstm_timesteps = 2
learning_rate = 1e-3

checkpoint_dir += r"/%s" % model_name
tensorboard_dir = checkpoint_dir + r"/tensorboard/test_him2017"


# load dataset
HIM2017 = dataset_loaders.HIM2017Loader()


# # PREPARE SESSION

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


# # LOAD MODEL

print("\n###   Loading Model: %s   ###\n" % model_name)
epoch_str = "Last epoch" if epoch is None else ("Epoch %i" % epoch)
print("##   %s   ##\n" % epoch_str)

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
    collection = tf.get_collection(COLLECTION)
    if len(collection) == 6:
        input_tensor, output_tensor, groundtruth_tensor, train_step, loss, merged = collection
        is_training = tf.placeholder(tf.bool)
    elif len(collection) == 7:
        input_tensor, output_tensor, groundtruth_tensor, train_step, loss, merged, is_training = collection
    else:
        raise ValueError("Expected 6 or 7 tensors in tf.get_collection(COLLECTION), but found %i:\n%r"
                         % (len(collection), collection))

    test_image_batch_gen = re.image_batch_generator_one_epoch(HIM2017.test_root,
                                                              HIM2017.test_list,
                                                              re.Const.TEST_BATCH_SIZE,
                                                              progress_desc="Collecting network output",
                                                              leave=True)
    predictions = None
    global_step = 0
    for batch in test_image_batch_gen:
        X = batch
        actual_batch_size = X.shape[0]
        X = X.reshape([actual_batch_size, *input_shape[1:]])
        Y_dummy = np.zeros([actual_batch_size, 21, 3])  # necessary as the restored "merged" tensor computes the loss

        # POTENTIAL ERRORS: this script assumes that MHP models use flattened output (63), wheres non-MHP models use
        # separate output dimensions per joint (21, 3). If an error arises when validating a model, check here.
        if len(collection) == 7:
            Y_dummy = Y_dummy.reshape([actual_batch_size, 63])

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

# results are in mm in every dimension

npyname = predictions_npy_name(model_name, epoch)
print("Storing prediction results at %s …" % npyname)

np.save(tensorboard_dir + "/" + npyname, predictions)

print("Done, exiting.")
exit(0)


# The following is mostly for debugging or quickly seeing the mean error

# # CALCULATE RESULTS

# mean absolute error

print("Calculating errors …")

# get ground truth labels
validate_label_gen = re.sample_generator(HIM2017.validate_root, "pose", HIM2017.validate_list)
validate_label = np.asarray(list(validate_label_gen))
# reshape from [set_size, 63] to [set_size, 21, 3]
validate_label = np.reshape(validate_label, [-1, *output_shape[-2:]])

# each individual error [ validate_set_length, 21, 3 ]
individual_error = np.abs(validate_label - predictions)

# overall error
overall_mean_error = ErrorCalculator.overall_mean_error(individual_error)

print("\n# %s" % epoch_str)
print("Mean prediction error (euclidean):", overall_mean_error)

pred_joint_avg = np.mean(predictions, axis=0)
actual_joint_avg = np.mean(validate_label, axis=0)

print("Actual joints position average:\n%r" % actual_joint_avg)
print("Predicted joints position average:\n%r" % pred_joint_avg)

print("Validation done.")
print("Point tensorboard to %s to get more insights. This directory also holds the .npy files for error analysis."
      % tensorboard_dir)
