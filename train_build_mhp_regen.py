# build and train a MHP extended network from scratch

import region_ensemble.model as re
import multiple_hypotheses_extension as mhp
from helpers import *

import tensorflow as tf
import keras.backend as K

import os


# dataset path declarations

prefix = "train-mhp-regen02"
checkpoint_dir = r"/home/matthias-k/GraphLSTM_data/%s" % prefix

dataset_root = r"/home/matthias-k/datasets/hands2017/data/hand2017_nor_img_new"
train_and_validate_list = ["nor_%08d.pkl" % i for i in range(1000, 957001, 1000)] + ["nor_00957032.pkl"]

train_list, validate_list = train_validate_split(train_and_validate_list)

# number of hypotheses produced by the extended regen network
hypotheses_count = 5

learning_rate = 1e-3

model_name = "regen-nopca_MHP_%ihyps_adamlr%f" % (hypotheses_count, learning_rate)

checkpoint_dir += r"/%s" % model_name
tensorboard_dir = checkpoint_dir + r"/tensorboard"


# # PREPARE SESSION

# set Keras session
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


# # BUILD MODEL

print("\n###   Building Model: %s   ###\n" % model_name)

print("Building RegionEnsemble network …")

# initialize region_ensemble_net
# region_ensemble_net_pca = re.RegEnPCA(directory_prefix=prefix, use_precalculated_samples=False,
#                                       dataset_root=dataset_root, train_list=train_and_validate_list)
# ... without the final layer
region_ensemble_net = re.RegEnModel(directory_prefix=prefix, result_layer=False)
# not sure if this works, as weights in the method are addressed by [-1]
# region_ensemble_net.set_pca_bottleneck_weights(region_ensemble_net_pca)

regen_input_tensor = region_ensemble_net.input
regen_40dim_output_tensor = region_ensemble_net.output

# build multiple hypotheses stage

print("Building Multiple Hypotheses stage …")

groundtruth_shape = [None, 63]
groundtruth_tensor = tf.placeholder(tf.float32, shape=groundtruth_shape)
is_training = tf.placeholder(tf.bool)

kernel_regularizer = tf.contrib.layers.l2_regularizer(re.Const.WEIGHT_DECAY)

mhp_input = regen_40dim_output_tensor
loss_func = re.soft_loss
mhp_layer, meta_loss = mhp.dense_mhp(mhp_input,
                                     units=63,
                                     hypotheses_count=hypotheses_count,
                                     loss_func=loss_func,
                                     is_training=is_training,
                                     groundtruth_tensor=groundtruth_tensor,
                                     kernel_regularizer=kernel_regularizer,
                                     name="dense_%i-hypotheses_MHP_layer" % hypotheses_count)

# MHP output tensor reshaped to [ batch_size, hypothesis_count, 21, 3 ]
mhp_layer_21_3 = tf.reshape(mhp_layer, shape=[-1, hypotheses_count, 21, 3])

# here the MHP Region Ensemble network is done initialising

print("Finished building model.\n")


# # TRAIN

print("Preparing training …")

max_epoch = 100
start_epoch = 1

input_shape = region_ensemble_net.input_shape  # = [None, *re.Const.MODEL_IMAGE_SHAPE]
input_tensor = regen_input_tensor

output_shape = [None, hypotheses_count, 21, 3]
output_tensor = mhp_layer_21_3

loss = meta_loss
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    print("Created new checkpoint directory `%s`." % checkpoint_dir)
saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.1, filename=checkpoint_dir)

# gather tensors for tensorboard
tf.summary.scalar('loss', loss)
# tf.summary.histogram('Graph LSTM output', glstm_output)
# tf.summary.histogram('Region Ensemble net output', regen_output_tensor_reshaped)
tf.summary.histogram('Network output', output_tensor)
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)
    print("Created new tensorboard directory `%s`." % tensorboard_dir)
merged = tf.summary.merge_all()
training_summary_writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)

# gather tensors needed for resuming after loading
tf.add_to_collection(COLLECTION, input_tensor)
tf.add_to_collection(COLLECTION, output_tensor)
tf.add_to_collection(COLLECTION, groundtruth_tensor)
tf.add_to_collection(COLLECTION, train_step)
tf.add_to_collection(COLLECTION, loss)
tf.add_to_collection(COLLECTION, merged)
tf.add_to_collection(COLLECTION, is_training)

t = TQDMHelper()

print("Initializing variables …")

sess.run(tf.global_variables_initializer())

with sess.as_default():
    print("Saving model meta graph …")
    saver.export_meta_graph(filename=checkpoint_dir + "/%s.meta" % model_name)

    print("Starting training.")

    global_step = 0
    for epoch in range(start_epoch, max_epoch + 1):
        t.start()
        # if augmentation should happen: pass augmented=True
        training_sample_generator = re.pair_batch_generator_one_epoch(dataset_root, train_list,
                                                                      re.Const.TRAIN_BATCH_SIZE,
                                                                      shuffle=True, progress_desc="Epoch %i" % epoch,
                                                                      leave=False, epoch=epoch - 1)

        for batch in training_sample_generator:
            X, Y = batch
            actual_batch_size = X.shape[0]
            X = X.reshape([actual_batch_size, *input_shape[1:]])
            Y = Y.reshape([actual_batch_size, *groundtruth_shape[1:]])

            _, loss_value, summary = sess.run([train_step, loss, merged], feed_dict={input_tensor: X,
                                                                                     groundtruth_tensor: Y,
                                                                                     K.learning_phase(): 1,
                                                                                     is_training: True})

            training_summary_writer.add_summary(summary, global_step=global_step)
            global_step += 1
            t.write("Current loss: %f" % loss_value)

        t.stop()
        print("Training loss after epoch %i: %f" % (epoch, loss_value))
        if epoch < 5 or epoch % 5 == 0:
            saver.save(sess, save_path=checkpoint_dir + "/%s" % model_name, global_step=epoch)

print("Training done, exiting.")
print("For validation, run: python validate.py %s %s [<epoch>]" % (prefix, model_name))
