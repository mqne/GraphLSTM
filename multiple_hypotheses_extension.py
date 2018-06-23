import tensorflow as tf
from tensorflow.python.framework import tensor_shape

# kernel_regularizer = tf.contrib.layers.l2_regularizer(re.Const.WEIGHT_DECAY)
# loss_func = re.soft_loss
# ground_truth = tf.placeholder ## todo
#
# hyp_input = regen_40dim_output_tensor

HYPOTHESES_AXIS = 1


class DenseMultipleHypothesesLayer(tf.layers.Layer):
    """Tensorflow's densely-connected layer class, extended with MHP.

    Parameters:
        units: (int) Number of units in each of the dense layers.
        hypotheses_count: (int) How many hypotheses to produce. Determines the
          HYPOTHESES_AXIS dimension (default: second) of output.
        loss_func: (callable) The loss function to be used in the meta loss. Will be
          called like this: loss_func(groundtruth_tensor, hypothesis_output)
        is_training: (boolean tf tensor) Replaces K.learning_phase(). If True,
          the meta loss will be calculated. If False, groundtruth_tensor is
          not evaluated and can thus be set to a dummy value.
        groundtruth_tensor: (tf tensor) The tensor used for evaluating the loss.
        epsilon: (float, default: 0.05) Epsilon value used in the calculation of
          the soft Kronecker delta.
        p_dropout: (float, default: 0.01) The probability for dropping out a
          hypothesis in the calculation of the meta loss.
        kernel_regularizer: (defaults to None) Gets passed on to the
          tf.layers.dense layers.
        name: (string, defauls to None) Gets passed to the superclass constructor.
          Defines the name of the tensorflow layer.

    Returns:
        A tuple (mhp_layer, meta_loss), where
          mhp_layer has the hypotheses stacked along axis HYPOTHESIS_AXIS, and
          meta_loss is the meta loss.
    """

    def __init__(self, units, hypotheses_count, loss_func, is_training, groundtruth_tensor,
                 epsilon=0.05, p_dropout=0.01, kernel_regularizer=None, name=None):
        super().__init__(name=name)
        self._units = units
        self._hypotheses_count = hypotheses_count
        self._loss_func = loss_func
        self._is_training = is_training
        self._groundtruth_tensor = groundtruth_tensor
        self._epsilon = epsilon
        self._p_dropout = p_dropout
        self._kernel_regularizer = kernel_regularizer

    # build self._hypotheses_count dense layers, each with self._units units
    def build_hypotheses_layer(self, inputs):
        hyps = []
        for i in range(self._hypotheses_count):
            hyp = tf.layers.dense(inputs, units=self._units, kernel_regularizer=self._kernel_regularizer)
            hyps.append(hyp)

        return hyps

    # soft Kronecker delta array
    def kds(self, index, m, length=None):
        """Return a vector populated by epsilon at index index, and epsilon / (m - 1).

        If length is given, it will be the length of the returned vector,
        with the additional elements equaling epsilon / (m - 1) as well.
        This is useful for dropout via multiplication by 0.
        """
        if length is None:
            length = m
        kd_1 = 1 - self._epsilon
        kd_0 = self._epsilon / tf.to_float(tf.maximum(m, 2) - 1)
        sparse = tf.scatter_nd([[index]], [kd_1 - kd_0], [length])
        return kd_0 + sparse

    # return n predictions stacked along axis HYPOTHESES_AXIS = 1 as well as the meta loss
    def call(self, inputs):
        """return signature: output, loss

        Uses the keras K.learning_phase() flag
        """

        hyps = self.build_hypotheses_layer(inputs)

        output = tf.stack(hyps, axis=HYPOTHESES_AXIS)

        return output, tf.cond(self._is_training,
                               true_fn=lambda: self._meta_loss(hyps),
                               false_fn=lambda: -1.)

    # meta loss used for learning
    def _meta_loss(self, hyps):
        # calculate losses for each hypothesis
        hyps_losses_all = [self._loss_func(self._groundtruth_tensor, h) for h in hyps]

        # store the number of actual hypotheses.
        # This should be equal to self._hypotheses_count if the layer has been initialized normally.
        h_length = len(hyps_losses_all)

        # calculate how many hypothesis to randomly leave out ("drop out")
        n_dropout = tf.to_int32(tf.log(tf.random_uniform([])) / tf.log(self._p_dropout))

        # not allowing less than 1 hypothesis
        n_hyps = tf.maximum(h_length - n_dropout, 1)

        # calculate pseudo loss that is worse than any existing loss
        bad_loss = tf.reduce_max(hyps_losses_all) + 1

        # determine which hypotheses to drop.
        # Note: as indices are randomly sampled, hypotheses may be chosen more than once to be dropped, yet the overall
        # amount of chosen indices remains < the number of hypotheses. Dropping a high number n of hypotheses
        # simultaneously is thus less likely than just p_dropout^n. As these probabilities are very low either way,
        # and a, in absolute terms, slightly lower probability of dropping many hypotheses in the same step does not
        # influence the overall result significantly, this detail is ignored for the sake of easier implementation.
        # Implementation via tf.random_shuffle does not work, as that function does not preserve gradient information.
        drop_indices = tf.random_uniform([h_length - n_hyps, 1],
                                         maxval=h_length, dtype=tf.int32)

        # add one dummy entry so that we don't have to care about an empty dropout array
        drop_indices_incl_dummy = tf.concat([drop_indices, [[h_length]]], axis=0)
        hyps_losses_all.append(0)

        # make sure the best hypothesis is chosen from a non-dropped loss by increasing to-be-dropped losses
        hyps_losses_all += tf.scatter_nd(drop_indices_incl_dummy,
                                         tf.squeeze(tf.ones_like(drop_indices_incl_dummy, dtype=tf.float32) * bad_loss, axis=1),
                                         [h_length + 1])

        # get index of best hypothesis
        min_loss_index = tf.argmin(hyps_losses_all, output_type=tf.int32)

        # now set dropped losses to 0
        hyps_losses_all *= tf.scatter_nd(drop_indices_incl_dummy,
                                         tf.squeeze(tf.zeros_like(drop_indices_incl_dummy, dtype=tf.float32), axis=1) - 1,
                                         [h_length + 1]
                                         ) + 1

        # weigh each loss by the soft kronecker delta via vector of kronecker deltas self.kds()
        weighted_losses = hyps_losses_all * self.kds(min_loss_index, n_hyps, h_length + 1)

        # sum over weighted losses and return
        return tf.reduce_sum(weighted_losses)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if input_shape[-1].value is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        one_layer_input_shape = input_shape[:-1].concatenate(self._units)
        return one_layer_input_shape[:HYPOTHESES_AXIS].concatenate(self._hypotheses_count).concatenate(
            one_layer_input_shape[HYPOTHESES_AXIS:])


def dense_mhp(inputs, units, hypotheses_count, loss_func, is_training, groundtruth_tensor,
              epsilon=0.05, p_dropout=0.01, kernel_regularizer=None, name=None):
    """Functional interface for the densely-connected layer extended with MHP.

    Parameters:
        inputs: Tensor input.
        units: (int) Number of units in each of the dense layers.
        hypotheses_count: (int) How many hypotheses to produce. Determines the
          HYPOTHESES_AXIS dimension (default: second) of output.
        loss_func: (callable) The loss function to be used in the meta loss. Will be
          called like this: loss_func(groundtruth_tensor, hypothesis_output)
        is_training: (boolean tf tensor) Replaces K.learning_phase(). If True,
          the meta loss will be calculated. If False, groundtruth_tensor is
          not evaluated and can thus be set to a dummy value.
        groundtruth_tensor: (tf tensor) The tensor used for evaluating the loss.
        epsilon: (float, default: 0.05) Epsilon value used in the calculation of
          the soft Kronecker delta.
        p_dropout: (float, default: 0.01) The probability for dropping out a
          hypothesis in the calculation of the meta loss.
        kernel_regularizer: (defaults to None) Gets passed on to the
          tf.layers.dense layers.
        name: (string, defauls to None) Gets passed to the superclass constructor.
          Defines the name of the tensorflow layer.

    Returns:
        A tuple (mhp_layer, meta_loss)
    """

    layer = DenseMultipleHypothesesLayer(units,
                                         hypotheses_count,
                                         loss_func,
                                         is_training,
                                         groundtruth_tensor,
                                         epsilon=epsilon,
                                         p_dropout=p_dropout,
                                         kernel_regularizer=kernel_regularizer,
                                         name=name)
    # quick way to make the input 'forget' about its keras nature, which otherwise creates weird issues
    inputs = tf.multiply(inputs, 1.)
    return layer.apply(inputs)


# calculate mean and variance across all hypotheses
def mean_and_variance(input_tensor):
    """return signature: mean, variance (across all hypotheses; input gets reduced along dimension 1"""
    return tf.nn.moments(input_tensor, axes=[HYPOTHESES_AXIS])


# return individual hypotheses as separate tensors
def individual_hypotheses(input_tensor):
    """unstacks the input along the hypotheses dimension"""
    return tf.unstack(input_tensor, axis=HYPOTHESES_AXIS)
