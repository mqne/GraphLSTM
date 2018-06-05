import graph_lstm as glstm
import region_ensemble.model as re

import tensorflow as tf
import keras.backend as K
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, TensorBoard

import numpy as np
import seaborn as sns
import zipfile

import os


# set plot style
# sns.set_style("whitegrid")


# dataset path declarations

prefix = "train-01"
checkpoint_dir = r"/data2/GraphLSTM/%s" % prefix

dataset_root = r"/mnt/nasbi/shared/research/hand-pose-estimation/hands2017/data/hand2017_nor_img_new"
train_list = ["nor_%08d.pkl" % i for i in range(1000, 957001, 1000)] + ["nor_00957032.pkl"]
validate_list = []

testset_root = r"/mnt/nasbi/shared/research/hand-pose-estimation/hands2017/data/hand2017_test_0914"
test_list = ["%08d.pkl" % i for i in range(10000, 290001, 10000)] + ["00295510.pkl"]


# 21 joint hand graph as used in hands2017 dataset
HAND_GRAPH_HANDS2017 = [("TMCP", "Wrist"), ("IMCP", "Wrist"), ("MMCP", "Wrist"), ("RMCP", "Wrist"), ("PMCP", "Wrist"),
                        ("IMCP", "MMCP"), ("MMCP", "RMCP"), ("RMCP", "PMCP"),
                        ("TMCP", "TPIP"), ("TPIP", "TDIP"), ("TDIP", "TTIP"),
                        ("IMCP", "IPIP"), ("IPIP", "IDIP"), ("IDIP", "ITIP"),
                        ("MMCP", "MPIP"), ("MPIP", "MDIP"), ("MDIP", "MTIP"),
                        ("RMCP", "RPIP"), ("RPIP", "RDIP"), ("RDIP", "RTIP"),
                        ("PMCP", "PPIP"), ("PPIP", "PDIP"), ("PDIP", "PTIP")]

# joint order as used in hands2017 dataset
HAND_GRAPH_HANDS2017_INDEX_DICT = {"Wrist": 0,
                                   "TMCP": 1, "IMCP": 2, "MMCP": 3, "RMCP": 4, "PMCP": 5,
                                   "TPIP": 6, "TDIP": 7, "TTIP": 8,
                                   "IPIP": 9, "IDIP": 10, "ITIP": 11,
                                   "MPIP": 12, "MDIP": 13, "MTIP": 14,
                                   "RPIP": 15, "RDIP": 16, "RTIP": 17,
                                   "PPIP": 18, "PDIP": 19, "PTIP": 20}


# # BUILD MODEL

# set Keras session
config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


print("Building RegionEnsemble network …")

# initialize region_ensemble_net
region_ensemble_net_pca = re.RegEnPCA(directory_prefix=prefix, use_precalculated_samples=True,
                                      dataset_root=dataset_root, train_list=train_list)
region_ensemble_net = re.RegEnModel(directory_prefix=prefix)
region_ensemble_net.compile(optimizer=Adam(), loss=re.soft_loss)
region_ensemble_net.set_pca_bottleneck_weights(region_ensemble_net_pca)

regen_input_tensor = region_ensemble_net.input
regen_output_tensor = region_ensemble_net.output
# here the Region Ensemble network is done initialising


print("Building GraphLSTM network …")

# initialize Graph LSTM
# since a well-defined node order is necessary to correctly communicate with the Region Ensemble network,
# the graph must be created manually
nxgraph = glstm.GraphLSTMNet.create_nxgraph(HAND_GRAPH_HANDS2017, num_units=3,
                                            index_dict=HAND_GRAPH_HANDS2017_INDEX_DICT)
graph_lstm_net = glstm.GraphLSTMNet(nxgraph, shared_weights=glstm.NEIGHBOUR_CONNECTIONS_SHARED)

# input dimensions of GraphLSTMNet: batch_size, max_time, number_of_nodes, input_size
regen_output_tensor_plus_timedim = tf.reshape(regen_output_tensor, shape=[-1, 1, 21, 3])  # todo replace numbers by references, move to graph_lstm_net.reshape_input

glstm_output_and_state = tf.nn.dynamic_rnn(graph_lstm_net, inputs=regen_output_tensor_plus_timedim, dtype=tf.float32)  # todo batch size
glstm_output = glstm_output_and_state[0]
# here the Graph LSTM network is done initializing

print("Finished building model.\n")

# # TRAIN

print("Preparing training …")

model_name = "regen41_graphlstm1"

max_epoch = 10
start_epoch = 1

input_shape = region_ensemble_net.input_shape  # = [None, *re.Const.MODEL_IMAGE_SHAPE]
input_tensor = regen_input_tensor

output_shape = [None, len(graph_lstm_net.output_size), graph_lstm_net.output_size[0]]
output_tensor = glstm_output

groundtruth_tensor = tf.placeholder(tf.float32, shape=output_shape)

train_step = tf.train.AdamOptimizer().minimize(re.soft_loss(groundtruth_tensor, output_tensor))

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    print("Created new directory `%s`." % checkpoint_dir)

saver = tf.train.Saver(keep_checkpoint_every_n_hours=1, filename=checkpoint_dir)

print("Initializing variables …")

sess.run(tf.global_variables_initializer())

print("Starting training.")

with sess.as_default():
    for epoch in range(start_epoch, max_epoch + 1):
        training_sample_generator = re.pair_batch_generator_one_epoch(dataset_root, train_list,
                                                                      re.Const.TRAIN_BATCH_SIZE,
                                                                      shuffle=True, progress_desc="Epoch %i" % epoch,
                                                                      leave=True, epoch=epoch - 1)
        for batch in training_sample_generator:
            X, Y = batch
            actual_batch_size = X.shape[0]
            X = X.reshape([actual_batch_size, *input_shape[1:]])
            Y = Y.reshape([actual_batch_size, *output_shape[1:]])

            sess.run(train_step, feed_dict={input_tensor: X, groundtruth_tensor: Y, K.learning_phase(): 1})

            # todo: pass K.learning_phase(): 1 to feed_dict (for testing: 0)
        saver.save(sess, save_path=checkpoint_dir + "/%s" % model_name, global_step=epoch)

print("Training done, exiting.")
exit(0)

train_batch_gen = re.pair_batch_generator(dataset_root, train_list, re.Const.TRAIN_BATCH_SIZE, shuffle=True, augmented=True)
validate_batch_gen = re.pair_batch_generator(dataset_root, validate_list, re.Const.VALIDATE_BATCH_SIZE)

history = region_ensemble_net.fit_generator(
    train_batch_gen,
    steps_per_epoch=re.Const.NUM_TRAIN_BATCHES,
    epochs=200,
    initial_epoch=30,
    callbacks=[
        #         LearningRateScheduler(lr_schedule),
        TensorBoard(log_dir="./%s" % region_ensemble_net.directory_prefix),
        ModelCheckpoint(
            filepath='./%s/region_ensemble_net.{epoch:02d}.hdf5' % region_ensemble_net.directory_prefix,
        ),
    ]
)


# # Load Weights

region_ensemble_net.load_weights('./%s/region_ensemble_net.30.hdf5' % prefix)  # , custom_objects={'soft_loss': soft_loss})


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


# # Explore Test

test_model_image_gen = re.sample_generator(testset_root, "image", test_list, resize_to_shape=re.Const.MODEL_IMAGE_SHAPE)
test_src_image_gen = re.sample_generator(testset_root, "image", test_list)

test_model_image = next(test_model_image_gen)
test_src_image = next(test_src_image_gen)
test_uvd = region_ensemble_net.predict(np.asarray([test_model_image]))
re.plot_scatter2d(test_src_image, pred=test_uvd)
re.plot_scatter3d(test_src_image, pred=test_uvd)


# # Test All

test_image_batch_gen = re.image_batch_generator(testset_root, test_list, re.Const.TEST_BATCH_SIZE)
# test_params is already loaded

test_uvd = region_ensemble_net.predict_generator(
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
                   in re.test_xyz_and_name_gen(region_ensemble_net, testset_root=testset_root, test_list=test_list))

np.savetxt('./%s/result-newtest.txt' % prefix, np.asarray(list(pose_submit_gen)), delimiter='\n', fmt="%s")

with zipfile.ZipFile("./%s/result-newtest.zip" % prefix, 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write("./%s/result-newtest.txt" % prefix, "result.txt")

