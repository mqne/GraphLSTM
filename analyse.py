from helpers import *
import plot_helper

import numpy as np
from matplotlib import pyplot as plt

from datetime import datetime
import os

plt.switch_backend('agg')

# prefix, model_name, epoch = get_prefix_model_name_optionally_epoch()

# syntax of file: each line containing one file.npy - label pair, separated by ','
file_with_npy_paths = get_from_commandline_args(1, "path to text file containing .npy filenames")[0]


# dataset path declarations

groundtruth_npy_location = r"/home/matthias/validate_split0.8_groundtruth.npy"

# checkpoint_dir = r"/home/matthias-k/GraphLSTM_data/%s" % prefix

dataset_root = r"/home/matthias-k/datasets/hands2017/data/hand2017_nor_img_new"
train_and_validate_list = ["nor_%08d.pkl" % i for i in range(1000, 957001, 1000)] + ["nor_00957032.pkl"]

train_list, validate_list = train_validate_split(train_and_validate_list)

testset_root = r"/data2/datasets/hands2017/data/hand2017_test_0914"
test_list = ["%08d.pkl" % i for i in range(10000, 290001, 10000)] + ["00295510.pkl"]

# number of timesteps to be simulated (each step, the same data is fed)
graphlstm_timesteps = 2
learning_rate = 1e-3

HYPOTHESES_AXIS = 1

# checkpoint_dir += r"/%s" % model_name
# tensorboard_dir = checkpoint_dir + r"/tensorboard/validation"


def clean_read_names_and_labels_from_file(path):
    with open(path) as f:
        lines = f.read().splitlines()
    if len(lines) == 0:
        raise LookupError("File seems to contain no entries.")
    combined_list = list(filter(lambda item: item.strip(), lines))
    combined_list = [s.split(',') for s in combined_list if not s.startswith(('#',))]
    if len(combined_list) == 0:
        raise LookupError("File seems to contain no entries.")
    names, labels = zip(*combined_list)
    return tuple([s.strip() for s in names]), tuple([s.strip() for s in labels])


save_path = datetime.utcnow().strftime("analysis_%y%m%d-%H%M%S/")
if not os.path.exists(save_path):
    os.makedirs(save_path)


# # LOAD PREDICTIONS

print("Loading predictions …", end=' ')

npy_names, npy_labels = clean_read_names_and_labels_from_file(file_with_npy_paths)

predictions_list = [np.load(file) for file in npy_names]

ndims_list = [np.ndim(a) for a in predictions_list]

if ndims_list[0] not in {3, 4}:
    print("\nERROR: first npy array expected to be of 3 or 4 dimensions, but found %i" % ndims_list[0])
    exit(1)

is_mhp = ndims_list[0] == 4

for ndim in ndims_list[1:]:
    if ndim != ndims_list[0]:
        print("\nERROR: Not all npy arrays of dimensionality %i (dimensionality of first array)." % ndims_list[0])
        exit(1)

print("done, %i loaded (%s)." % (len(predictions_list), ('MHP' if is_mhp else 'regular')))

# if not server:
#     tensorboard_dir = "/home/matthias/predictions"
# else:
#     predictions_filename = predictions_npy_name(model_name, epoch)
# predictions = np.load(tensorboard_dir + "/" + predictions_filename)


# # GROUND TRUTH

try:
    print("Loading ground truth …", end=' ')
    groundtruth = np.load(groundtruth_npy_location)
except FileNotFoundError:
    print("Calculating ground truth and storing into %s …" % groundtruth_npy_location, end=' ')
    groundtruth = calc_groundtruth_poses_npy(dataset_root, validate_list)
    np.save(groundtruth_npy_location, groundtruth)

print("done.")


# # CALCULATE RESULTS

# mean absolute error

print("Calculating errors …")

# everything is a list of results over each npy results array, dim: [number of npy files, ... ]

if is_mhp == 4:

    p_mean_list, p_variance_list = zip(*[np_mean_and_variance(predictions) for predictions in predictions_list])
    p_std_list = [np.sqrt(v) for v in p_variance_list]

    individual_hypotheses_errors_list = [np.abs(predictions - np.expand_dims(groundtruth, axis=HYPOTHESES_AXIS))
                                         for predictions in predictions_list]

    average_std_per_joint_list = [np.mean(ps, axis=(0, HYPOTHESES_AXIS)) for ps in p_std_list]

    # list of index orders suitable for feeding into helpers.confidence_dict_for_index_order
    joint_indices_ranked_by_std_list = [np.argsort(a) for a in average_std_per_joint_list]

else:

    # each individual error [ validate_set_length, 21, 3 ]
    individual_errors_list = [np.abs(predictions - groundtruth) for predictions in predictions_list]

    # hands2017 error measure 1
    mean_errors = [ErrorCalculator.overall_mean_error(x) for x in individual_errors_list]
    print("\nMean overall error:")
    for label, mean_error in zip(npy_labels, mean_errors):
        print(str(mean_error) + "\t" + label)

    print("\nCreating plots …")
    # hands2017 error measure 2
    plot_helper.plot_ratio_of_joints_within_bound(individual_errors_list,
                                                  legend=npy_labels,
                                                  savepath=save_path+"measure2")

    # hands2017 error measure 3
    plot_helper.plot_ratio_of_frames_with_all_joints_within_bound(individual_errors_list,
                                                                  legend=npy_labels,
                                                                  savepath=save_path+"measure3")

    if len(npy_names) == 1:
        plot_helper.violinplot_error_per_joint(individual_errors_list[0],
                                               group_by_joint_type=True,
                                               max_err=10,
                                               savepath=save_path+"joint_violin_"+os.path.basename(npy_names[0]))

    print("Plots stored at %s." % save_path)

print("Done.")
exit(0)
