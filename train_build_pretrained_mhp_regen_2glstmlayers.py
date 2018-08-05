# build and train a two layer Graph LSTM network on top of a pre-trained MHP architecture

import graph_lstm as glstm
import region_ensemble.model as re
import multiple_hypotheses_extension as mhp
from helpers import *

import tensorflow as tf
import keras.backend as K

import os


# pretrained model paths

load_epoch = 92  # ADAPT HERE
hypotheses_count = 2

# index update order, created by analysis.py of RegEn MHP 2 hypotheses model at epoch 92
index_order_confidence_regen_mhp2 = [3, 5, 4, 15, 16, 12, 13, 9, 2, 6, 1, 7, 0, 10, 18, 14, 17, 19, 11, 8, 20]
index_order_wrist_first = [0, 1, 2, 3, 4, 5, 6, 9, 12, 15, 18, 7, 10, 13, 16, 19, 8, 11, 14, 17, 20]
index_order_tips_first = list(reversed(index_order_wrist_first))
# reverse for performance evaluation
# index_order_confidence_regen_mhp2.reverse()

glstm_confidence_dict = confidence_dict_for_index_order(index_order_tips_first)

pretrained_prefix = "train-mhp-regen02"
pretrained_model_name = "regen-nopca_MHP_%ihyps_adamlr0.001000" % hypotheses_count


# dataset path declarations

prefix = "train-pr-mhp-regen02-gl2"
checkpoint_dir = r"/home/matthias-k/GraphLSTM_data/%s" % prefix

# checkpoint dir of the pretrained model
pretrained_checkpoint_dir = r"/home/matthias-k/GraphLSTM_data/%s/%s" % \
                            (pretrained_prefix, pretrained_model_name)

dataset_root = r"/home/matthias-k/datasets/hands2017/data/hand2017_nor_img_new"
train_and_validate_list = ["nor_%08d.pkl" % i for i in range(1000, 957001, 1000)] + ["nor_00957032.pkl"]

train_list, validate_list = train_validate_split(train_and_validate_list)

# number of timesteps to be simulated (each step, the same data is fed)
graphlstm_timesteps = 2
learning_rate = 1e-3

model_name = "regen_MHP%ihyps_pretrained_epoch%i_lrx0.1_graphlstmt%i_rescon_2glstmlayers_adamlr%f" % \
             (hypotheses_count, load_epoch, graphlstm_timesteps, learning_rate)

checkpoint_dir += r"/%s" % model_name
tensorboard_dir = checkpoint_dir + r"/tensorboard"


# # PREPARE SESSION

# set Keras session
config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


# # BUILD MODEL

print("\n###   Building Model: %s   ###\n" % model_name)

print("Loading MHP-RegionEnsemble network …")

with sess.as_default():
    with tf.variable_scope("Pretrained_MHP_RegEn_network"):
        print("> Loading meta graph …")
        loader = tf.train.import_meta_graph(pretrained_checkpoint_dir + "/%s.meta" % pretrained_model_name)
        print("> Restoring weights for epoch %i …" % load_epoch)
        loader.restore(sess, pretrained_checkpoint_dir + "/%s-%i" % (pretrained_model_name, load_epoch))
        print("> Getting necessary tensors …")
        pm_input_tensor, pm_output_tensor, pm_groundtruth_tensor, pm_train_step, pm_loss, pm_merged, pm_is_training = \
            tf.get_collection(COLLECTION)
        # clear collection of stored references from graph
        tf.get_default_graph().clear_collection(COLLECTION)

# input of pretrained model is input tensor of RegEn net
regen_input_tensor = pm_input_tensor
# output of pretrained model is output tensor of RegEn net reshaped to [batch_size, hypotheses_count, 21, 3]
regen_output_tensor_reshaped_mhp = pm_output_tensor
is_training = pm_is_training

print("Preparing MHP output for GraphLSTM …")

# set learning rate of RegEn network to 0.1 that of GLSTM
regen_output_tensor_reshaped_mhp = lr_mult(0.1)(regen_output_tensor_reshaped_mhp)

# store set of pretrained weights to be excluded from variable initialisation
pretrained_variables = set(tf.global_variables())

# get mean and variance of MHP network
mhp_mean, mhp_variance = mhp.mean_and_variance(regen_output_tensor_reshaped_mhp)

mhp_one_output = mhp_mean


print("Building GraphLSTM network …")

# FIRST LAYER

# initialize Graph LSTM
# since a well-defined node order is necessary to correctly communicate with the Region Ensemble network,
# the graph must be created manually
nxgraph = glstm.GraphLSTMNet.create_nxgraph(HAND_GRAPH_HANDS2017,
                                            num_units=GLSTM_NUM_UNITS,
                                            index_dict=HAND_GRAPH_HANDS2017_INDEX_DICT)
graph_lstm_net = glstm.GraphLSTMNet(nxgraph, shared_weights=glstm.NEIGHBOUR_CONNECTIONS_SHARED, name="GLSTM_layer_1")

# not necessary, as pretrained model already has output shape [ batch size, 21, 3 ]
# # overall output dimensions: batch_size, number_of_nodes, output_size
# regen_output_tensor_reshaped = graph_lstm_net.reshape_input_for_dynamic_rnn(regen_output_tensor)

# normalize RegEn net output
mhp_one_output_tensor_normalized, undo_scaling = normalize_for_glstm(mhp_one_output)


# input dimensions of GraphLSTMNet: batch_size, max_time, number_of_nodes, input_size
graphlstm_input_tensor = graph_lstm_net.reshape_input_for_dynamic_rnn(mhp_one_output_tensor_normalized,
                                                                      timesteps=graphlstm_timesteps)

dynrnn_glstm_output_full, dynrnn_glstm_state = tf.nn.dynamic_rnn(graph_lstm_net,
                                                                 inputs=graphlstm_input_tensor,
                                                                 dtype=tf.float32)

# reorder dimensions to [batch_size, max_time, number_of_nodes, output_size]
glstm_output_full = graph_lstm_net.transpose_output_from_cells_first_to_batch_first(dynrnn_glstm_output_full)
# extract last timestep from output
glstm_output = tf.unstack(glstm_output_full, axis=1)[-1]
# denormalize GraphLSTM output
glstm_output_rescaled = undo_scaling(glstm_output)

# here the Graph LSTM network is done initializing

# Graph LSTM output is in range (-1, 1) because of tanh in the hidden state
# employ residual connections around Graph LSTM

residual_merge_1 = tf.add(mhp_one_output, glstm_output_rescaled)


# SECOND LAYER

# initialize Graph LSTM
# since a well-defined node order is necessary to correctly communicate with the Region Ensemble network,
# the graph must be created manually
nxgraph_2 = glstm.GraphLSTMNet.create_nxgraph(HAND_GRAPH_HANDS2017,
                                              num_units=GLSTM_NUM_UNITS,
                                              index_dict=HAND_GRAPH_HANDS2017_INDEX_DICT)
graph_lstm_net_2 = glstm.GraphLSTMNet(nxgraph_2, shared_weights=glstm.NEIGHBOUR_CONNECTIONS_SHARED,
                                      name="GLSTM_layer_2")

# not necessary, as pretrained model already has output shape [ batch size, 21, 3 ]
# # overall output dimensions: batch_size, number_of_nodes, output_size
# regen_output_tensor_reshaped = graph_lstm_net.reshape_input_for_dynamic_rnn(regen_output_tensor)

# normalize RegEn net output
residual_merge_1_normalized, undo_scaling_rm1 = normalize_for_glstm(residual_merge_1)


# input dimensions of GraphLSTMNet: batch_size, max_time, number_of_nodes, input_size
graphlstm_input_tensor_2 = graph_lstm_net_2.reshape_input_for_dynamic_rnn(residual_merge_1_normalized,
                                                                          timesteps=graphlstm_timesteps)

dynrnn_glstm_output_full_2, dynrnn_glstm_state_2 = tf.nn.dynamic_rnn(graph_lstm_net_2,
                                                                     inputs=graphlstm_input_tensor_2,
                                                                     dtype=tf.float32)

# reorder dimensions to [batch_size, max_time, number_of_nodes, output_size]
glstm_output_full_2 = graph_lstm_net_2.transpose_output_from_cells_first_to_batch_first(dynrnn_glstm_output_full_2)
# extract last timestep from output
glstm_output_2 = tf.unstack(glstm_output_full_2, axis=1)[-1]
# denormalize GraphLSTM output
glstm_output_rescaled_2 = undo_scaling_rm1(glstm_output_2)

# here the Graph LSTM network is done initializing

# Graph LSTM output is in range (-1, 1) because of tanh in the hidden state
# employ residual connections around Graph LSTM

residual_merge_2 = tf.add(residual_merge_1, glstm_output_rescaled_2)

print("Finished building model.\n")


# # TRAIN

print("Preparing training …")

max_epoch = 100
start_epoch = 1

input_shape = [None, *re.Const.MODEL_IMAGE_SHAPE]
input_tensor = regen_input_tensor

output_shape = [None, len(graph_lstm_net_2.output_size), graph_lstm_net_2.output_size[0]]
output_tensor = residual_merge_2

groundtruth_tensor = tf.placeholder(tf.float32, shape=output_shape)

loss = re.soft_loss(groundtruth_tensor, output_tensor)
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate, name="Adam_%s" % model_name).minimize(loss)

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    print("Created new checkpoint directory `%s`." % checkpoint_dir)
saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.1, filename=checkpoint_dir)

# gather tensors for tensorboard
s_loss = tf.summary.scalar('loss', loss)
s_glstm_output = tf.summary.histogram('Graph LSTM_2 output', glstm_output_2)
s_regen_output = tf.summary.histogram('MHP Region Ensemble net one output', mhp_one_output)
s_network_output = tf.summary.histogram('Network output', output_tensor)
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)
    print("Created new tensorboard directory `%s`." % tensorboard_dir)
merged = tf.summary.merge([s_loss, s_glstm_output, s_regen_output, s_network_output])
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

print("Initializing new variables …")

# initialise all variables minus those loaded with the pretrained RegEn net
sess.run(tf.variables_initializer(set(tf.global_variables()) - pretrained_variables))

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
            Y = Y.reshape([actual_batch_size, *output_shape[1:]])

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
