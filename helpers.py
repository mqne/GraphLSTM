from sys import stdout, argv
from tqdm import tqdm
import tensorflow as tf
import numpy as np


# tensorflow collection name for saving important tensors
COLLECTION = "input-output-groundtruth-trainstep-loss"

# each cell has three units, one per dimension (x, y, z)
GLSTM_NUM_UNITS = 3


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


# camera parameters for the hands2017 dataset
FX_HANDS2017 = 475.065948
FY_HANDS2017 = 475.065857
CX_HANDS2017 = 315.944855
CY_HANDS2017 = 245.287079


def train_validate_split(train_validate_list, split=0.8):
    cut = int(len(train_validate_list) * split)
    return train_validate_list[:cut], train_validate_list[cut:]


def normalize_for_glstm(tensor):  # todo move to GraphLSTMNet?
    # this function assumes tensors with shape [ batch_size, number_of_nodes, output_size=3 ]
    assert(len(tensor.shape) == 3)
    # compute maximum and minimum joint position value in each of x,y,z
    max_dim = tf.reduce_max(tensor, axis=1, keepdims=True)
    min_dim = tf.reduce_min(tensor, axis=1, keepdims=True)
    diff_dim = tf.subtract(max_dim, min_dim)
    # get normalizing factor as maximum difference within all 3 dimensions
    max_diff = tf.reduce_max(diff_dim, axis=2, keepdims=True)
    normalized_tensor = tf.divide(tensor - min_dim - diff_dim / 2, max_diff)

    # return output rescaled and shifted to original position
    def unnormalize(tensor):
        return tf.multiply(tensor, max_diff) + diff_dim / 2 + min_dim

    # return output only rescaled, centered around 0
    def undo_scaling(tensor):
        return tf.multiply(tensor, max_diff)

    return normalized_tensor, undo_scaling


# learning rate multiplier for tensorflow >= 1.8
# as suggested by user1735003 at https://stackoverflow.com/a/50388264
def lr_mult(alpha):
    """Usage: lr_mult(multiplier)(tensor)
    """
    @tf.custom_gradient
    def _lr_mult(x):
        def grad(dy):
            return dy * alpha * tf.ones_like(x)
        return x, grad
    return _lr_mult


def get_from_commandline_args(count, args_string=None):
    if len(argv) - 1 != count:
        print("You need to enter exactly %i command line arguments%s, "
              "but found %i" % (count, '' if args_string is None else "(%s)" % args_string, len(argv) - 1))
        exit(1)

    return tuple(argv[1:])


# get prefix and model name from command line
def get_prefix_and_model_name():
    return get_from_commandline_args(2, "'prefix' and 'model_name'")


# get prefix, model name and epoch from command line
def get_prefix_model_name_and_epoch():
    r = get_from_commandline_args(3, "'prefix', 'model_name' and epoch")
    return r[0], r[1], int(r[2])


# get prefix, model name and optionally epoch from command line
def get_prefix_model_name_optionally_epoch():
    count = len(argv) - 1
    if count == 2:
        return (*get_prefix_and_model_name(), None)
    elif count == 3:
        return get_prefix_model_name_and_epoch()
    else:
        print("You need to enter either 2 or 3 command line arguments ('prefix', 'model_name' and optionally epoch), "
              "but found %i" % count)
        exit(1)


# create namestring for saving predictions array given model name and epoch
def predictions_npy_name(model_name, epoch):
    return "predictions_%s%s.npy" % (model_name, (("_epoch" + str(epoch)) if epoch is not None else ""))


def calc_groundtruth_poses_npy(dataset_root, container_name_list):
    """Returns all groundtruth poses for the given dataset and container name list.
    """
    from region_ensemble.model import sample_generator as s
    labels_63 = np.asarray(list(s(dataset_root, "pose", container_name_list)))
    labels_21_3 = np.reshape(labels_63, [-1, 21, 3])
    return labels_21_3


def uvd3xyz(uvd, fx=FX_HANDS2017, fy=FY_HANDS2017, cx=CX_HANDS2017, cy=CY_HANDS2017):
    """Convert a pose batch of dimensions (batch_size, joints, 3)
    from uvd to xyz (mm).
    """

    uvd = np.asarray(uvd)
    assert uvd.ndim == 3
    assert uvd.shape[-1] == 3

    f = np.asarray([1/fx, 1/fy, 1])
    cc1 = np.asarray([-cx, -cy, 1])
    uv0 = uvd * np.asarray([1, 1, 0])
    d = np.expand_dims(uvd[:, :, -1], 2)
    xyz = d * f * (uv0 + cc1)
    return xyz


def uvd2xyz(uvd, fx=FX_HANDS2017, fy=FY_HANDS2017, cx=CX_HANDS2017, cy=CY_HANDS2017):
    """Convert a single pose of dimensions (joints, 3)
    from uvd to xyz (mm)."""
    uvd = np.asarray(uvd)
    assert uvd.ndim == 2
    return uvd3xyz(np.expand_dims(uvd, 0), fx=fx, fy=fy, cx=cx, cy=cy)[0]


class ErrorCalculator:

    @staticmethod
    def reduce_xyz_norm(individual_errors):
        return np.linalg.norm(individual_errors, axis=2)

    @staticmethod
    # error averaged over joints and dimensions [ variable_set_length ]
    def per_frame(individual_errors):
        return np.mean(ErrorCalculator.reduce_xyz_norm(individual_errors), axis=1)

    @staticmethod
    # error averaged over dimensions [ variable_set length, 21 ]
    def per_frame_and_joint(individual_errors):
        return ErrorCalculator.reduce_xyz_norm(individual_errors)

    @staticmethod
    # joint and dimension error averaged over all samples [ 21, 3 ]
    def per_joint_dim(individual_errors):
        return np.mean(individual_errors, axis=0)

    @staticmethod
    # joint error averaged over samples and dimensions [ 21 ]
    def per_joint(individual_errors):
        return np.mean(ErrorCalculator.reduce_xyz_norm(individual_errors), axis=0)

    @staticmethod
    # dimension error averaged over samples and joints [ 3 ]
    def per_dimension(individual_errors):
        return np.mean(individual_errors, axis=(0, 1))

    @staticmethod
    # overall error
    def overall_mean_error_dimensions_averaged(individual_errors):
        return np.mean(individual_errors)

    @staticmethod
    # overall error
    def overall_mean_error_euclidean(individual_errors):
        return np.mean(ErrorCalculator.reduce_xyz_norm(individual_errors))


class TQDMHelper:
    """For printing additional info while a tqdm bar is active.

    Example:
        t = TQDMHelper()
        t.start()  # create newline to not overwrite previous output
        for x in range(2):
            g = tqdm_decorated_generator()
            for i in g:
                t.write("Status message for %i" % i)  # write/update message
                sleep(.05)
        t.stop()  # clear line
    """
    def __init__(self):
        from tqdm._utils import _term_move_up, _environ_cols_wrapper
        self._r_prefix = _term_move_up() + '\r'
        self._dynamic_ncols = _environ_cols_wrapper()

    def _write_raw(self, message):
        tqdm.write(self._r_prefix + message)

    def _clear(self):
        ncols = None
        if self._dynamic_ncols:
            ncols = self._dynamic_ncols(stdout)
        if ncols is None:
            ncols = 50
        self._write_raw(" " * ncols)

    @staticmethod
    def start():
        print()

    def write(self, message):
        self._clear()
        self._write_raw(message)

    def stop(self):
        self._clear()
