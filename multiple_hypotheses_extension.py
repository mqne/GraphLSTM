import tensorflow as tf
from tensorflow.python.framework import tensor_shape
import keras.backend as K

kernel_regularizer = tf.contrib.layers.l2_regularizer(re.Const.WEIGHT_DECAY)
loss_func = re.soft_loss
ground_truth = tf.placeholder ## todo

hyp_input = regen_40dim_output_tensor

HYPOTHESES_AXIS = 1


class DenseMultipleHypothesesLayer(tf.layers.Layer):

    def __init__(self, units, hypotheses_count, loss_func, epsilon=0.05, p_dropout=0.01, kernel_regularizer=None, name=None):
        super().__init__(name=name)
        self._units = units
        self._hypotheses_count = hypotheses_count
        self._loss_func = loss_func
        self._epsilon = epsilon
        self._p_dropout = p_dropout
        self._kernel_regularizer = kernel_regularizer

    def build_hypotheses_layer(self, inputs):
        hyps = []
        for i in range(self._hypotheses_count):
            hyp = tf.layers.dense(inputs, units=self._units, kernel_regularizer=self._kernel_regularizer)
            hyps.append(hyp)

        return hyps

    # soft Kronecker delta
    def kd(self, a):
        if a:
            return 1 - self._epsilon
        return self._epsilon / (self._hypotheses_count - 1)

    # return n predictions stacked along axis HYPOTHESES_AXIS = 1 as well as the meta loss
    # TODO: implement dropout in training
    def call(self, inputs):
        """return signature: output, loss

        Uses the keras K.learning_phase() flag
        """

        hyps = self.build_hypotheses_layer(inputs)

        output = tf.stack(hyps, axis=HYPOTHESES_AXIS)

        if K.learning_phase() == 0:
            return output, None

        hyps_losses = [self._loss_func(ground_truth, h) for h in hyps]

        min_loss_index = tf.argmin(hyps_losses)

        weighted_losses = [self.kd(i == min_loss_index) * hyps_losses[i] for i in range(len(hyps))]

        return output, tf.reduce_sum(weighted_losses)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if input_shape[-1].value is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        one_layer_input_shape = input_shape[:-1].concatenate(self._units)
        return one_layer_input_shape[0].concatenate(self._hypotheses_count).concatenate(one_layer_input_shape[1:])


def mean_and_variance(input_tensor):
    """return signature: mean, variance (across all hypotheses; input gets reduced along dimension 1"""
    return tf.nn.moments(input_tensor, axes=[HYPOTHESES_AXIS])


def individual_hypotheses(input_tensor):
    """unstacks the input along the hypotheses dimension"""
    return tf.unstack(input_tensor, axis=HYPOTHESES_AXIS)

