# plots qualitative plots of predicted and ground truth hand poses on top of corresponding depth maps

import region_ensemble.model as re
from helpers import *

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import Bbox
import plot_helper


dataset_root = r"/mnt/HDD_data/hands2017/data/hand2017_nor_img_new"
train_and_validate_list = ["nor_%08d.pkl" % i for i in range(1000, 957001, 1000)] + ["nor_00957032.pkl"]

train_list, validate_list = train_validate_split(train_and_validate_list)

prediction_root = "/home/matthias/predictions"
gt_location = "/home/matthias/validate_split0.8_groundtruth.npy"

glstm_pred_name = "predictions_regen41_pretrained_epoch90_lrx0.1_graphlstm1bf1t2_rescon_adamlr0.001000_epoch90.npy"
dpren_pred_name = "predictions_regen41_adamlr0.001000_epoch90.npy"

# load ground truth
gt_npy = np.load(gt_location)

# load model predictions
glstm_pr_npy = np.load(prediction_root + "/" + glstm_pred_name)
dpren_pr_npy = np.load(prediction_root + "/" + dpren_pred_name)


def npy_generator(npy):
    for pose in npy:
        yield pose


def edges_from_pose(pose):
    # yields dimensions [edge_count (23), coordinate_count (3), nodes_per_edge (2)]
    return np.array([(pose[HAND_GRAPH_HANDS2017_INDEX_DICT[edge[0]]],
                      pose[HAND_GRAPH_HANDS2017_INDEX_DICT[edge[1]]])
                     for edge in HAND_GRAPH_HANDS2017]).swapaxes(1, 2)


def plt_scatter2d(image=None, prediction=None, groundtruth=None,
                  figlen=plot_helper.PAGEWIDTH_INCHES/3, marker_size_factor=1.0, filename=None,
                  pred_colour=plot_helper.TumColours.Blue, gt_colour=plot_helper.TumColours.XGrey_65):

    if image is None and prediction is None and groundtruth is None:
        print("You need to specify at least one out of image, prediction, groundtruth for plt_scatter2d")
        return

    prediction = prediction.reshape([21, 3]) if prediction is not None else None
    groundtruth = groundtruth.reshape([21, 3]) if groundtruth is not None else None

    colour_map = LinearSegmentedColormap.from_list('grey_to_white',
                                                   ((0.611764705882353, 0.615686274509804, 0.6235294117647059),
                                                    (1, 1, 1)))
    figsize = (figlen, figlen)
    plt.figure(num=None, figsize=figsize)
    fig = plt.subplot(aspect='equal')

    plt.xlim(0, re.Const.SRC_IMAGE_SHAPE[0])
    plt.ylim(re.Const.SRC_IMAGE_SHAPE[1], 0)

    if image is not None:
        image = np.squeeze(image)
        image[image <= 0] = np.max(image) + 10

        plt.imshow(image, extent=(0, re.Const.SRC_IMAGE_SHAPE[0], re.Const.SRC_IMAGE_SHAPE[1], 0), cmap=colour_map, zorder=-10)

    if prediction is not None:
        pr = prediction.swapaxes(0, 1)
        plt.scatter(pr[0], pr[1], c=pred_colour, marker='.', alpha=0.8, zorder=-2,
                    s=plt.rcParamsDefault['lines.markersize'] ** 2 * 2 * marker_size_factor, linewidths=0.0)
        for edge in edges_from_pose(prediction):
            plt.plot(edge[0], edge[1], c=pred_colour, alpha=0.8, zorder=-3)

    if groundtruth is not None:
        gt = groundtruth.swapaxes(0, 1)
        plt.scatter(gt[0], gt[1], c=gt_colour, marker='.', alpha=0.8, zorder=-5,
                    s=plt.rcParamsDefault['lines.markersize'] ** 2 * 2 * marker_size_factor, linewidths=0.0)
        for edge in edges_from_pose(groundtruth):
            plt.plot(edge[0], edge[1], c=gt_colour, alpha=0.8, zorder=-6)

    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    if filename is not None:
        bbox = Bbox(((0, 0), (figsize[0], figsize[1])))
        plt.savefig("/tmp/qualitative_hand_plots/%s.pdf" % filename, bbox_inches=bbox, dpi=300)
        plt.savefig("/tmp/qualitative_hand_plots/%s.png" % filename, bbox_inches=bbox, dpi=300)
    # plt.show()
    plt.close()


src_image_gen = re.sample_generator(dataset_root, "image", validate_list)
gt_gen = npy_generator(gt_npy)
glstm_pr_gen = npy_generator(glstm_pr_npy)
dpren_pr_gen = npy_generator(dpren_pr_npy)

# print("Press Return to advance to next image.")
i = 0
for img, gt, pr_dpren, pr_glstm in zip(src_image_gen, gt_gen, dpren_pr_gen, glstm_pr_gen):
    # re.plot_scatter3d(img, pr, gt)
    # plt_scatter2d(None, pr, None, filename='pred_small_%i_dpren' % i, figlen=1, marker_size_factor=1/1.5)
    plt_scatter2d(img, pr_dpren, gt, filename='4_per_line/pred_gt_%i_dpren' % i, figlen=plot_helper.PAGEWIDTH_INCHES*0.24, marker_size_factor=0.8)
    plt_scatter2d(img, pr_glstm, gt, filename='4_per_line/pred_gt_%i_glstm' % i, pred_colour=plot_helper.TumColours.AccentOrange, figlen=plot_helper.PAGEWIDTH_INCHES/4, marker_size_factor=0.8)
    # plt_scatter2d(img, None, gt, filename='gt_%i' % i)
    print(i)
    i += 1
    # exit(0)
    if i >= 30:
        break
