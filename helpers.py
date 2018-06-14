from sys import stdout, argv
from tqdm import tqdm
import tensorflow as tf


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


def get_prefix_model_name_and_epoch():
    r = get_from_commandline_args(3, "'prefix', 'model_name' and epoch")
    return r[0], r[1], int(r[2])


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
