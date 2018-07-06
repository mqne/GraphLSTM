from helpers import *
import plot_helper

import numpy as np
from matplotlib import pyplot as plt

from datetime import datetime
import os

plt.switch_backend('agg')

# prefix, model_name, epoch = get_prefix_model_name_optionally_epoch()


# get prefix, model name and optionally epoch from command line
def get_npyfilename_and_optional_switch():
    count = len(argv) - 1
    s = "path to text file containing .npy filenames (and optional switch for overall accuracy curve)"
    if count == 1:
        return (*get_from_commandline_args(1, s), None)
    elif count == 2:
        return get_from_commandline_args(2, s)
    else:
        print("You need to enter either 1 or 2 command line arguments (path to text file containing .npy filenames "
              "and optionally do_overall_plot switch), but found %i" % count)
        exit(1)


# syntax of file: each line containing one file.npy - label pair, separated by ','
file_with_npy_paths, do_overall_plot = get_npyfilename_and_optional_switch()

do_overall_plot = do_overall_plot is not None

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
if do_overall_plot:
    save_path = save_path.replace('_', '_overall_')
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

if not do_overall_plot:

    for ndim in ndims_list[1:]:
        if ndim != ndims_list[0]:
            print("\nERROR: Not all npy arrays of dimensionality %i (dimensionality of first array)." % ndims_list[0])
            exit(1)

print("done, %i loaded (%s)." % (len(predictions_list), ('overall' if do_overall_plot
                                                         else ('MHP' if is_mhp else 'regular'))))

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


def analyse_regular(individual_errors_list, labels, prefix='', colours=None):
    # hands2017 error measure 1
    mean_errors = [ErrorCalculator.overall_mean_error(x) for x in individual_errors_list]
    with open(save_path + prefix + 'mean_overall_errors.txt', 'a') as f:
        print("\n" + prefix + "Mean overall error:")
        for label, mean_error in zip(labels, mean_errors):
            s = str(mean_error) + "\t" + label
            f.write(s + '\n')
            print(s)
    print("\nCreating plots …")
    # hands2017 error measure 2
    plot_helper.plot_ratio_of_joints_within_bound(individual_errors_list,
                                                  legend=labels,
                                                  savepath=save_path + prefix + "measure2",
                                                  colours=colours)
    # hands2017 error measure 3
    plot_helper.plot_ratio_of_frames_with_all_joints_within_bound(individual_errors_list,
                                                                  legend=labels,
                                                                  savepath=save_path + prefix + "measure3",
                                                                  colours=colours)


def hyp_colour_gen():
    for c in [plot_helper.TumColours.LightGray, plot_helper.TumColours.Gray, plot_helper.TumColours.DarkGray]:
        yield c


def regular_colour_gen():
    for c in [plot_helper.TumColours.AccentOrange, plot_helper.TumColours.Blue, plot_helper.TumColours.AccentBlue,
              plot_helper.TumColours.SecondaryBlue2]:
        yield c


if do_overall_plot:

    regular_error_label_list = []
    mhp_error_label_list = []
    mhp_mean_error_label_list = []

    for i, prediction in enumerate(predictions_list):
        if prediction.ndim == 3:
            regular_error_label_list.append((np.abs(prediction - groundtruth),
                                             npy_labels[i]))
        elif prediction.ndim == 4:
            mhp_error_label_list.append((np.abs(prediction - np.expand_dims(groundtruth, axis=HYPOTHESES_AXIS)),
                                         npy_labels[i]))
            mhp_mean_error_label_list.append((np.abs(np.mean(prediction, axis=HYPOTHESES_AXIS) - groundtruth),
                                              npy_labels[i]))
        else:
            raise ValueError("Found incompatible dimensions (neither 3 nor 4) in %s: %i (shape: %r)" %
                             (npy_names[i], prediction.ndim, prediction.shape))

    if len(regular_error_label_list) > len(list(regular_colour_gen())):
        raise ValueError("Too many regular predictions for currently implemented number of colours")
    if len(mhp_error_label_list) > len(list(hyp_colour_gen())):
        raise ValueError("Too many MHP predictions for currently implemented number of colours")

    print("Plotting full MHP results …")

    prefix = 'full_MHP_'

    plot_labels = []
    plot_errors = []
    plot_colours = []
    hyp_colours = hyp_colour_gen()
    regular_colours = regular_colour_gen()

    print("\nUserWarnings considering the label '_nolegend_' are expected and can be ignored.\n")

    for mhp_mean, mhp_label in mhp_error_label_list:
        colour = next(hyp_colours)
        for i, hyp in enumerate(mhp_mean.swapaxes(0, 1)):
            if i == 0:
                plot_labels.append(mhp_label)
            else:
                plot_labels.append('_nolegend_')
            plot_errors.append(hyp)
            plot_colours.append(colour)

    for reg, reg_label in regular_error_label_list:
        colour = next(regular_colours)
        plot_labels.append(reg_label)
        plot_errors.append(reg)
        plot_colours.append(colour)

    # hands2017 error measure 2
    plot_helper.plot_ratio_of_joints_within_bound(plot_errors,
                                                  legend=plot_labels,
                                                  savepath=save_path + prefix + "measure2",
                                                  colours=plot_colours)
    # hands2017 error measure 3
    plot_helper.plot_ratio_of_frames_with_all_joints_within_bound(plot_errors,
                                                                  legend=plot_labels,
                                                                  savepath=save_path + prefix + "measure3",
                                                                  colours=plot_colours)

    print("Plotting mean MHP results …")

    prefix = 'mean_MHP_'

    plot_labels = []
    plot_errors = []
    plot_colours = []
    hyp_colours = hyp_colour_gen()
    regular_colours = regular_colour_gen()

    for mhp_mean, mhp_label in mhp_mean_error_label_list:
        colour = next(hyp_colours)
        plot_labels.append("Mean " + mhp_label)
        plot_errors.append(mhp_mean)
        plot_colours.append(colour)

    for reg, reg_label in regular_error_label_list:
        colour = next(regular_colours)
        plot_labels.append(reg_label)
        plot_errors.append(reg)
        plot_colours.append(colour)

    # hands2017 error measure 2
    plot_helper.plot_ratio_of_joints_within_bound(plot_errors,
                                                  legend=plot_labels,
                                                  savepath=save_path + prefix + "measure2",
                                                  colours=plot_colours)
    # hands2017 error measure 3
    plot_helper.plot_ratio_of_frames_with_all_joints_within_bound(plot_errors,
                                                                  legend=plot_labels,
                                                                  savepath=save_path + prefix + "measure3",
                                                                  colours=plot_colours)

elif is_mhp:

    p_mean_list, p_variance_list = zip(*[np_mean_and_variance(predictions) for predictions in predictions_list])
    # p_std_list = [np.sqrt(np.abs(v)) for v in p_variance_list]

    individual_hypotheses_errors_list = [np.abs(predictions - np.expand_dims(groundtruth, axis=HYPOTHESES_AXIS))
                                         for predictions in predictions_list]

    average_var_per_joint_list = [np.mean(ps, axis=(0, 2)) for ps in p_variance_list]

    # list of index orders suitable for feeding into helpers.confidence_dict_for_index_order
    joint_indices_ranked_by_std_list = [np.argsort(a) for a in average_var_per_joint_list]

    with open(save_path+'joint_indices_ranked_by_confidence_(high_to_low).txt', 'a') as f:
        print("\nJoint indices ranked from high to low confidence:")
        for label, indices in zip(npy_labels, joint_indices_ranked_by_std_list):
            s = label + '\n' + str(indices)
            f.write(s + '\n')
            print(s)

    with open(save_path + 'var_per_joint.txt', 'a') as f:
        reverse_index_dict = reverse_dict(HAND_GRAPH_HANDS2017_INDEX_DICT)
        for label, std_array in zip(npy_labels, average_var_per_joint_list):
            s = label
            f.write(s + '\n')
            for joint, std in zip([reverse_index_dict[i] for i in range(len(HAND_GRAPH_HANDS2017_INDEX_DICT))],
                                  std_array):
                s = joint + '\t' + str(std)
                f.write(s + '\n')
            f.write('\n')

    for label, label_errors in zip(npy_labels, individual_hypotheses_errors_list):
        label_errors = np.swapaxes(label_errors, 0, 1)
        analyse_regular(label_errors,
                        ['Hypothesis ' + str(i_hyp) for i_hyp in range(len(label_errors))],
                        prefix=label+'_',
                        colours=[plot_helper.TumColours.LightGray] * len(label_errors))


else:

    # each individual error [ validate_set_length, 21, 3 ]
    individual_errors_list = [np.abs(predictions - groundtruth) for predictions in predictions_list]

    analyse_regular(individual_errors_list, npy_labels)

    if len(npy_names) == 1:
        # y maximum: 2*max_err, rounded up to next 5mm increment
        max_avg_joint_err = np.max(ErrorCalculator.per_joint(individual_errors_list[0]))
        plot_helper.violinplot_error_per_joint(individual_errors_list[0],
                                               group_by_joint_type=True,
                                               max_err=((max_avg_joint_err*2)//5+1)*5,
                                               savepath=save_path + "joint_violin_" + os.path.basename(npy_names[0]))

print("\nPlots stored at %s." % save_path)

print("Done.")
exit(0)
