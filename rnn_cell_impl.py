# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Module implementing RNN Cells.

This module provides a number of basic commonly used RNN cells, such as LSTM
(Long Short Term Memory) or GRU (Gated Recurrent Unit), and a number of
operators that allow adding dropouts, projections, or embeddings for inputs.
Constructing multi-layer cells is supported by the class `MultiRNNCell`, or by
calling the `rnn` ops several times.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import hashlib
import numbers

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


def _like_rnncell(cell):
    """Checks that a given object is an RNNCell by using duck typing."""
    conditions = [hasattr(cell, "output_size"), hasattr(cell, "state_size"),
                  hasattr(cell, "zero_state"), callable(cell)]
    return all(conditions)


def _concat(prefix, suffix, static=False):
    """Concat that enables int, Tensor, or TensorShape values.

    This function takes a size specification, which can be an integer, a
    TensorShape, or a Tensor, and converts it into a concatenated Tensor
    (if static = False) or a list of integers (if static = True).

    Args:
      prefix: The prefix; usually the batch size (and/or time step size).
        (TensorShape, int, or Tensor.)
      suffix: TensorShape, int, or Tensor.
      static: If `True`, return a python list with possibly unknown dimensions.
        Otherwise return a `Tensor`.

    Returns:
      shape: the concatenation of prefix and suffix.

    Raises:
      ValueError: if `suffix` is not a scalar or vector (or TensorShape).
      ValueError: if prefix or suffix was `None` and asked for dynamic
        Tensors out.
    """
    if isinstance(prefix, ops.Tensor):
        p = prefix
        p_static = tensor_util.constant_value(prefix)
        if p.shape.ndims == 0:
            p = array_ops.expand_dims(p, 0)
        elif p.shape.ndims != 1:
            raise ValueError("prefix tensor must be either a scalar or vector, "
                             "but saw tensor: %s" % p)
    else:
        p = tensor_shape.as_shape(prefix)
        p_static = p.as_list() if p.ndims is not None else None
        p = (constant_op.constant(p.as_list(), dtype=dtypes.int32)
             if p.is_fully_defined() else None)
    if isinstance(suffix, ops.Tensor):
        s = suffix
        s_static = tensor_util.constant_value(suffix)
        if s.shape.ndims == 0:
            s = array_ops.expand_dims(s, 0)
        elif s.shape.ndims != 1:
            raise ValueError("suffix tensor must be either a scalar or vector, "
                             "but saw tensor: %s" % s)
    else:
        s = tensor_shape.as_shape(suffix)
        s_static = s.as_list() if s.ndims is not None else None
        s = (constant_op.constant(s.as_list(), dtype=dtypes.int32)
             if s.is_fully_defined() else None)

    if static:
        shape = tensor_shape.as_shape(p_static).concatenate(s_static)
        shape = shape.as_list() if shape.ndims is not None else None
    else:
        if p is None or s is None:
            raise ValueError("Provided a prefix or suffix of None: %s and %s"
                             % (prefix, suffix))
        shape = array_ops.concat((p, s), 0)
    return shape


def _zero_state_tensors(state_size, batch_size, dtype):
    """Create tensors of zeros based on state_size, batch_size, and dtype."""

    def get_state_shape(s):
        """Combine s with batch_size to get a proper tensor shape."""
        c = _concat(batch_size, s)
        c_static = _concat(batch_size, s, static=True)
        size = array_ops.zeros(c, dtype=dtype)
        size.set_shape(c_static)
        return size

    return nest.map_structure(get_state_shape, state_size)


class RNNCell(base_layer.Layer):
    """Abstract object representing an RNN cell.

    Every `RNNCell` must have the properties below and implement `call` with
    the signature `(output, next_state) = call(input, state)`.  The optional
    third input argument, `scope`, is allowed for backwards compatibility
    purposes; but should be left off for new subclasses.

    This definition of cell differs from the definition used in the literature.
    In the literature, 'cell' refers to an object with a single scalar output.
    This definition refers to a horizontal array of such units.

    An RNN cell, in the most abstract setting, is anything that has
    a state and performs some operation that takes a matrix of inputs.
    This operation results in an output matrix with `self.output_size` columns.
    If `self.state_size` is an integer, this operation also results in a new
    state matrix with `self.state_size` columns.  If `self.state_size` is a
    (possibly nested tuple of) TensorShape object(s), then it should return a
    matching structure of Tensors having shape `[batch_size].concatenate(s)`
    for each `s` in `self.batch_size`.
    """

    def __call__(self, inputs, state, scope=None):
        """Run this RNN cell on inputs, starting from the given state.

        Args:
          inputs: `2-D` tensor with shape `[batch_size x input_size]`.
          state: if `self.state_size` is an integer, this should be a `2-D Tensor`
            with shape `[batch_size x self.state_size]`.  Otherwise, if
            `self.state_size` is a tuple of integers, this should be a tuple
            with shapes `[batch_size x s] for s in self.state_size`.
          scope: VariableScope for the created subgraph; defaults to class name.

        Returns:
          A pair containing:

          - Output: A `2-D` tensor with shape `[batch_size x self.output_size]`.
          - New state: Either a single `2-D` tensor, or a tuple of tensors matching
            the arity and shapes of `state`.
        """
        if scope is not None:
            with vs.variable_scope(scope,
                                   custom_getter=self._rnn_get_variable) as scope:
                return super(RNNCell, self).__call__(inputs, state, scope=scope)
        else:
            with vs.variable_scope(vs.get_variable_scope(),
                                   custom_getter=self._rnn_get_variable):
                return super(RNNCell, self).__call__(inputs, state)

    def _rnn_get_variable(self, getter, *args, **kwargs):
        variable = getter(*args, **kwargs)
        trainable = (variable in tf_variables.trainable_variables() or
                     (isinstance(variable, tf_variables.PartitionedVariable) and
                      list(variable)[0] in tf_variables.trainable_variables()))
        if trainable and variable not in self._trainable_weights:
            self._trainable_weights.append(variable)
        elif not trainable and variable not in self._non_trainable_weights:
            self._non_trainable_weights.append(variable)
        return variable

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.

        It can be represented by an Integer, a TensorShape or a tuple of Integers
        or TensorShapes.
        """
        raise NotImplementedError("Abstract method")

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        raise NotImplementedError("Abstract method")

    def build(self, _):
        # This tells the parent Layer object that it's OK to call
        # self.add_variable() inside the call() method.
        pass

    def zero_state(self, batch_size, dtype):
        """Return zero-filled state tensor(s).

        Args:
          batch_size: int, float, or unit Tensor representing the batch size.
          dtype: the data type to use for the state.

        Returns:
          If `state_size` is an int or TensorShape, then the return value is a
          `N-D` tensor of shape `[batch_size x state_size]` filled with zeros.

          If `state_size` is a nested list or tuple, then the return value is
          a nested list or tuple (of the same structure) of `2-D` tensors with
          the shapes `[batch_size x s]` for each s in `state_size`.
        """
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            state_size = self.state_size
            return _zero_state_tensors(state_size, batch_size, dtype)


class BasicRNNCell(RNNCell):
    """The most basic RNN cell.

    Args:
      num_units: int, The number of units in the RNN cell.
      activation: Nonlinearity to use.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
       in an existing scope.  If not `True`, and the existing scope already has
       the given variables, an error is raised.
    """

    def __init__(self, num_units, activation=None, reuse=None):
        super(BasicRNNCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or math_ops.tanh

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        """Most basic RNN: output = new_state = act(W * input + U * state + B)."""
        output = self._activation(_linear([inputs, state], self._num_units, True))
        return output, output


class GRUCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None):
        super(GRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        """Gated recurrent unit (GRU) with nunits cells."""
        with vs.variable_scope("gates"):  # Reset gate and update gate.
            # We start with bias of 1.0 to not reset and not update.
            bias_ones = self._bias_initializer
            if self._bias_initializer is None:
                dtype = [a.dtype for a in [inputs, state]][0]
                bias_ones = init_ops.constant_initializer(1.0, dtype=dtype)
            value = math_ops.sigmoid(
                _linear([inputs, state], 2 * self._num_units, True, bias_ones,
                        self._kernel_initializer))
            r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
        with vs.variable_scope("candidate"):
            c = self._activation(
                _linear([inputs, r * state], self._num_units, True,
                        self._bias_initializer, self._kernel_initializer))
        new_h = u * state + (1 - u) * c
        return new_h, new_h


_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))


class LSTMStateTuple(_LSTMStateTuple):
    """Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.

    Stores two elements: `(c, h)`, in that order.

    Only used when `state_is_tuple=True`.
    """
    __slots__ = ()

    @property
    def dtype(self):
        (c, h) = self
        if c.dtype != h.dtype:
            raise TypeError("Inconsistent internal state: %s vs %s" %
                            (str(c.dtype), str(h.dtype)))
        return c.dtype


class BasicLSTMCell(RNNCell):
    """Basic LSTM recurrent network cell.

    The implementation is based on: http://arxiv.org/abs/1409.2329.

    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.

    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.

    For advanced models, please use the full @{tf.nn.rnn_cell.LSTMCell}
    that follows.
    """

    def __init__(self, num_units, forget_bias=1.0,
                 state_is_tuple=True, activation=None, reuse=None):
        """Initialize the basic LSTM cell.

        Args:
          num_units: int, The number of units in the LSTM cell.
          forget_bias: float, The bias added to forget gates (see above).
            Must set to `0.0` manually when restoring from CudnnLSTM-trained
            checkpoints.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  The latter behavior will soon be deprecated.
          activation: Activation function of the inner states.  Default: `tanh`.
          reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.  If not `True`, and the existing scope already has
            the given variables, an error is raised.

          When restoring from CudnnLSTM-trained checkpoints, must use
          CudnnCompatibleLSTMCell instead.
        """
        super(BasicLSTMCell, self).__init__(_reuse=reuse)
        if not state_is_tuple:
            logging.warn("%s: Using a concatenated state is slower and will soon be "
                         "deprecated.  Use state_is_tuple=True.", self)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or math_ops.tanh

    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        """Long short-term memory cell (LSTM).

        Args:
          inputs: `2-D` tensor with shape `[batch_size x input_size]`.
          state: An `LSTMStateTuple` of state tensors, each shaped
            `[batch_size x self.state_size]`, if `state_is_tuple` has been set to
            `True`.  Otherwise, a `Tensor` shaped
            `[batch_size x 2 * self.state_size]`.

        Returns:
          A pair containing the new hidden state, and the new state (either a
            `LSTMStateTuple` or a concatenated state, depending on
            `state_is_tuple`).
        """
        sigmoid = math_ops.sigmoid
        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = array_ops.split(value=state, num_or_size_splits=2, axis=1)

        concat = _linear([inputs, h], 4 * self._num_units, True)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)

        new_c = (
                c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
        new_h = self._activation(new_c) * sigmoid(o)

        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = array_ops.concat([new_c, new_h], 1)
        return new_h, new_state


class GraphLSTMCell(RNNCell):
    """Graph LSTM recurrent network cell.

    The implementation is an adaption of tensorflow's BasicLSTMCell
    and based on: https://arxiv.org/abs/1603.07063.

    This class is part of the Master thesis of Matthias Kuehne at the
    Chair for Computer Aided Medical Procedures & Augmented Reality,
    Technical University of Munich, Germany
    in cooperation with the
    Robotics Vision Lab,
    Nara Institute of Science and Technology, Japan.

    The implementation is work in progress.

    > We add forget_bias (default: 1) to the biases of the forget gate in order to
    > reduce the scale of forgetting in the beginning of the training.
    >
    > It does not allow cell clipping, a projection layer, and does not
    > use peep-hole connections: it is the basic baseline.
    >
    > For advanced models, please use the full @{tf.nn.rnn_cell.LSTMCell}
    > that follows.
    """

    def __init__(self, num_units, forget_bias=1.0,
                 state_is_tuple=True, activation=None, reuse=None):
        """Initialize the Graph LSTM cell.

        Args:
          num_units: int, The number of units in the Graph LSTM cell.
          forget_bias: float, The bias added to forget gates (see above).
            Must set to `0.0` manually when restoring from CudnnLSTM-trained
            checkpoints.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  The latter behavior will soon be deprecated.
          activation: Activation function of the inner states.  Default: `tanh`.
          reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.  If not `True`, and the existing scope already has
            the given variables, an error is raised.

          When restoring from CudnnLSTM-trained checkpoints, must use
          CudnnCompatibleLSTMCell instead.
        """
        super(GraphLSTMCell, self).__init__(_reuse=reuse)
        if not state_is_tuple:
            logging.warn("%s: Using a concatenated state is UNTESTED for this GraphLSTM implementation. "
                         "If it works, it is likely to be slower and will soon be "
                         "deprecated.  Use state_is_tuple=True.", self)
        self._num_units = num_units
        self._forget_bias = forget_bias  # this parameter is currently ignored
        self._state_is_tuple = state_is_tuple
        self._activation = activation or math_ops.tanh  # this parameter is currently ignored

    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state, neighbour_states):
        """Graph long short-term memory cell (GraphLSTM).

        Args:
          inputs: `2-D` tensor with shape `[batch_size x input_size]`.
          state: An `LSTMStateTuple` of state tensors, each shaped
            `[batch_size x self.state_size]`, if `state_is_tuple` has been set to
            `True`.  Otherwise, a `Tensor` shaped
            `[batch_size x 2 * self.state_size]`.
          neighbour_states: a list of n `LSTMStateTuples` of state tensors (m_j, h_j)

        Returns:
          A pair containing the new hidden state, and the new state (either a
            `LSTMStateTuple` or a concatenated state, depending on
            `state_is_tuple`).
        """
        sigmoid = math_ops.sigmoid
        tanh = math_ops.tanh

        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
            m_i, h_i = state
        else:
            m_i, h_i = array_ops.split(value=state, num_or_size_splits=2, axis=1)

        # "shared weight metrics Ufn for all nodes are learned to guarantee the spatial transformation
        # invariance and enable the learning with various neighbors": GraphLSTM cells have to be generalized to be able
        # to be applied to any random image superpixel region, whereas for hand pose estimation, we want each cell to
        # specialize on its joint

        # in the paper, all cells are generalized and thus do not need to know about the nature of their
        # neighbours. However, we want cells specifically trained for certain joint, so information about which
        # neighbouring cell belongs to which node might be interesting ... kind of a "hard wired" Graph LSTM
        # But: that's good! -> Own contribution, learn generic hand model / even learn individual hand sizes?
        # TODO: first implement regular Graph LSTM, then test, then hand-specific version

        # TODO: unit tests for GraphLSTMCell.call

        # extract two vectors of n ms and n hs from state vector of n (m,h) tuples
        m_j_all, h_j_all = zip(*neighbour_states)

        # IMPLEMENTATION DIFFERS FROM PAPER: in eq. (2) g^f_ij uses h_j,t regardless of if node j has been updated
        # already or not. Implemented here is h_j,t for non-updated nodes and h_j,t+1 for updated nodes
        # which both makes sense intuitively (most recent information)
        # and is more lightweight (no need to keep track of old states)

        # Eq. 1: averaged hidden states for neighbouring nodes h^-_{i,t}
        h_j_avg = math_ops.reduce_mean(h_j_all, axis=0)

        # define weight and bias names
        w_u = "W_u"
        w_f = "W_f"
        w_c = "W_c"
        w_o = "W_o"
        u_u = "U_u"
        u_f = "U_f"
        u_c = "U_c"
        u_o = "U_o"
        u_un = "U_un"
        u_fn = "U_fn"
        u_cn = "U_cn"
        u_on = "U_on"
        b_u = "b_u"
        b_f = "b_f"
        b_c = "b_c"
        b_o = "b_o"

        # Eq. 2
        # input gate
        # g_u = sigmoid ( f_{i,t+1} * W_u + h_{i,t} * U_u + h^-_{i,t} * U_{un} + b_u )
        g_u = sigmoid(_graphlstm_linear([w_u, u_u, u_un, b_u], [inputs, h_i, h_j_avg], self.output_size, True))
        # adaptive forget gate
        # g_fij = sigmoid ( f_{i,t+1} * W_f + h_{j,t} * U_fn + b_f ) for every neighbour j
        g_fij = [sigmoid(_graphlstm_linear([w_f, u_fn, b_f], [inputs, h_j], self.output_size, True)) for h_j in h_j_all]
        # forget gate
        # g_fi = sigmoid ( f_{i,t+1} * W_f + h_{i,t} * U_f + b_f )
        g_fi = sigmoid(_graphlstm_linear([w_f, u_f, b_f], [inputs, h_i], self.output_size, True,
                                         reuse_weights=[w_f, b_f]))
        # output gate
        # g_o = sigmoid ( f_{i,t+1} * W_o + h_{i,t} * U_o + h^-_{i,t} * U_{on} + b_o )
        g_o = sigmoid(_graphlstm_linear([w_o, u_o, u_on, b_o], [inputs, h_i, h_j_avg], self.output_size, True))
        # memory gate
        # g_c = tanh ( f_{i,t+1} * W_c + h_{i,t} * U_c + h^-_{i,t} * U_{cn} + b_c )
        g_c = tanh(_graphlstm_linear([w_c, u_c, u_cn, b_c], [inputs, h_i, h_j_avg], self.output_size, True))

        # new memory states
        # m_i_new = sum ( g_fij .* most recent state of each neighbouring node ) / number of neighbouring nodes ...
        #       ... + g_fi .* m_i + g_u .* g_c
        m_i_new = math_ops.reduce_mean([g * m_j for g, m_j in zip(g_fij, m_j_all)], axis=0) + g_fi * m_i + g_u * g_c

        # new hidden states
        # h_i_new = tanh ( g_o .* m_i_new )
        h_i_new = tanh(g_o * m_i_new)

        # Eq. 3 (return values)
        if self._state_is_tuple:
            new_state = LSTMStateTuple(m_i_new, h_i_new)
        else:
            new_state = array_ops.concat([m_i_new, h_i_new], 1)
        return h_i_new, new_state


# calculates terms like W * f + U * h + b
def _graphlstm_linear(weight_names, args,
                      output_size,
                      bias,
                      bias_initializer=None,
                      weight_initializer=None, reuse_weights=None):
    """Linear map: sum_i(args[i] * weights[i]), where weights[i] can be multiple variables.

    Args:
      weight_names: a string or list of strings
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_initializer: starting value to initialize the bias
        (default is all zeros).
      weight_initializer: starting value to initialize the weight.
      reuse_weights: a string or list of strings defining which weights should be reused.

    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices for each W[i] not to be reused.

    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
      LookupError: if a weight name specified to be reused does not appear in the list of weights.
    """
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if weight_names is None or (nest.is_sequence(weight_names) and not weight_names):
        raise ValueError("`weight_names` must be specified")
    if not nest.is_sequence(args):
        args = [args]
    if not nest.is_sequence(weight_names):
        weight_names = [weight_names]
    if reuse_weights is not None:
        if not nest.is_sequence(reuse_weights):
            reuse_weights = [reuse_weights]
        for w in reuse_weights:
            if w not in weight_names:
                raise LookupError("'%s' in `reuse_weights` not found in `weight_names`" % str(w))

    # for each variable in 'args' there needs to be exactly one in "weights", plus bias
    if bias:
        if len(weight_names) != len(args) + 1:
            raise ValueError("If `bias` is True, `weight_names` needs to be one element longer than `args`,"
                             " but found: %d and %d, respectively" % (len(weight_names), len(args)))
    else:
        if len(weight_names) != len(args):
            raise ValueError("If `bias` is False, `weight_names` and `args` need to be of the same length,"
                             " but found: %d and %d, respectively" % (len(weight_names), len(args)))

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    scope = vs.get_variable_scope()
    with vs.variable_scope(scope) as outer_scope:
        summands = []
        for i, x in enumerate(args):
            reuse = True if reuse_weights is not None and weight_names[i] in reuse_weights else None
            with vs.variable_scope(outer_scope, reuse=reuse):
                weight = vs.get_variable(
                    name=weight_names[i], shape=[x.get_shape()[1].value, output_size],
                    dtype=dtype,
                    initializer=weight_initializer)
            summands.append(math_ops.matmul(x, weight))
        res = math_ops.add_n(summands)
        if bias:
            reuse = True if reuse_weights is not None and weight_names[-1] in reuse_weights else None
            with vs.variable_scope(outer_scope, reuse=reuse) as inner_scope:
                inner_scope.set_partitioner(None)
                if bias_initializer is None:
                    bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
                b = vs.get_variable(
                    name=weight_names[-1], shape=[output_size],
                    dtype=dtype,
                    initializer=bias_initializer)
                res = nn_ops.bias_add(res, b)
        return res


import networkx as nx

# identifiers for node attributes
_CELL = "cell"
_CONFIDENCE = "confidence"
_INDEX = "index"


class GraphLSTMNet(RNNCell):
    """GraphLSTM Network composed of multiple simple cells.

    The implementation is an adaption of tensorflow's MultiRNNCell
    and based on: https://arxiv.org/abs/1603.07063

    This class is part of the Master thesis of Matthias Kuehne at the
    Chair for Computer Aided Medical Procedures & Augmented Reality,
    Technical University of Munich, Germany
    in cooperation with the
    Robotics Vision Lab,
    Nara Institute of Science and Technology, Japan.

    The implementation is work in progress."""

    def _cell(self, node, graph=None):
        """Return the GraphLSTMCell belonging to a node.

        Args:
          node: The node whose GraphLSTMCell object will be returned.
          graph: The graph from which the node will be extracted. Defaults to self._graph .
        """
        if graph is None:
            graph = self._graph
        elif not isinstance(graph, nx.classes.graph.Graph):
            raise TypeError(
                "graph must be a Graph of package networkx, but saw: %s." % graph)

        return graph.node[node][_CELL]

    def __init__(self, graph, state_is_tuple=True, name=None):
        """Create a Graph LSTM Network composed of a graph of GraphLSTMCells.

        Args:
          graph: networkx.Graph containing GraphLSTMCells
          state_is_tuple: If True, accepted and returned states are n-tuples, where
            `n = len(cells)`.  If False, the states are all
            concatenated along the column axis.  This latter behavior will soon be
            deprecated.

        Raises:
          ValueError: if graph is empty (not allowed), or at least one of the cells
            returns a state tuple but the flag `state_is_tuple` is `False`.
        """
        super(GraphLSTMNet, self).__init__(name=name)
        if not graph:
            raise ValueError("Must specify graph for GraphLSTMNet.")
        if not isinstance(graph, nx.classes.graph.Graph):
            raise TypeError(
                "graph must be a Graph of package networkx, but saw: %s." % graph)

        self._graph = graph
        self._state_is_tuple = state_is_tuple
        if not state_is_tuple:
            if any(nest.is_sequence(self._cell(n).state_size) for n in self._graph):
                raise ValueError("Some cells return tuples of states, but the flag "
                                 "state_is_tuple is not set.  State sizes are: %s"
                                 % str([self._cell(n).state_size for n in self._graph]))

        # TODO init weights (W*, U*, U*n, b*) here for global weights

    @property
    def state_size(self):
        if self._state_is_tuple:
            return tuple(self._cell(n).state_size for n in self._graph)
        else:
            return sum([self._cell(n).state_size for n in self._graph])

    @property
    def output_size(self):
        return sum(self._cell(n).output_size for n in self._graph)

    def zero_state(self, batch_size, dtype):
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            if self._state_is_tuple:
                return tuple(self._cell(n).zero_state(batch_size, dtype) for n in self._graph)
            else:
                # We know here that state_size of each cell is not a tuple and
                # presumably does not contain TensorArrays or anything else fancy
                return super(GraphLSTMNet, self).zero_state(batch_size, dtype)

    def call(self, inputs, state):
        """Run this multi-layer cell on inputs, starting from state.

        Args:
          inputs: a tuple of inputs for each graph node, where the index must correspond to the node attribute 'index'
          state: a tuple or tensor of states for each node
        """

        # check if input size matches expected size
        if len(inputs) is not self._graph.number_of_nodes():
            raise ValueError("Number of nodes in GraphLSTMNet input %d does not match number of graph nodes %d" %
                             (len(inputs), self._graph.number_of_nodes()))

        # check how _linear() gets its tf variables (generation vs. reusing)
        # and use that knowledge for U and other variables
        # SOLVED: normal LSTM cells are created each in their own scope, and _linear gets called only once by each cell
        # as a cell is not really an object that gets called, but a chain of ops that gets trained or evaluated
        # weights shared between cells (e.g. Ufn) and weights unique for each cell (e.g. Uf): how to handle?
        # ^this is for global Ufn local Un, which is NOT in the original paper! Paper: everything is global
        # TODO: all weights global, tf.AUTO_REUSE in cell? init all weights in net?

        new_states = [None] * self._graph.number_of_nodes()
        graph_output = [None] * self._graph.number_of_nodes()

        # iterate over cells in graph, starting with highest confidence value
        for node_name, node_obj in sorted(self._graph.nodes(data=True), key=lambda x: x[1][_CONFIDENCE], reverse=True):
            # TODO variable scope to include graphLSTM name/instance-id/or similar
            with vs.variable_scope("cell_%s" % node_name):  # TODO: variable scope here? in other places?
                # extract GraphLSTMCell object from graph node
                cell = node_obj[_CELL]
                # extract node index for state vector addressing
                i = node_obj[_INDEX]
                # extract state of current cell
                if self._state_is_tuple:
                    if not nest.is_sequence(state):
                        raise ValueError(
                            "Expected state to be a tuple of length %d, but received: %s" %
                            (len(self.state_size), state))
                    cur_state = state[i]
                else:
                    cur_state_pos = cell.state_size * i
                    cur_state = array_ops.slice(state, [0, cur_state_pos],
                                                [-1, cell.state_size])

                # extract and collect states of neighbouring cells
                neighbour_states_array = []
                for neighbour_name, neighbour_obj in nx.all_neighbors(self._graph, node_name):
                    n_i = neighbour_obj[_INDEX]
                    # use updated state if node has been visited
                    # TODO: think about giving old _and_ new states to node for 100% paper fidelity
                    if new_states[n_i] is not None:
                        n_state = new_states[n_i]
                    elif self._state_is_tuple:
                        n_state = state[n_i]
                    else:
                        n_state_pos = cell.state_size * n_i
                        n_state = array_ops.slice(state, [0, n_state_pos],
                                                  [-1, cell.state_size])
                    neighbour_states_array.append(n_state)
                # make immutable
                neighbour_states = tuple(neighbour_states_array)
                # extract input of current cell from input tuple
                cur_inp = inputs[i]
                # run current cell
                cur_output, new_state = cell(cur_inp, cur_state, neighbour_states)
                # store cell output and state in graph vector
                graph_output[i] = cur_output
                new_states[i] = new_state

        # pack results and return
        graph_output = tuple(graph_output)
        new_states = (tuple(new_states) if self._state_is_tuple else
                      array_ops.concat(new_states, 1))

        return graph_output, new_states


class LSTMCell(RNNCell):
    """Long short-term memory unit (LSTM) recurrent network cell.

    The default non-peephole implementation is based on:

      http://www.bioinf.jku.at/publications/older/2604.pdf

    S. Hochreiter and J. Schmidhuber.
    "Long Short-Term Memory". Neural Computation, 9(8):1735-1780, 1997.

    The peephole implementation is based on:

      https://research.google.com/pubs/archive/43905.pdf

    Hasim Sak, Andrew Senior, and Francoise Beaufays.
    "Long short-term memory recurrent neural network architectures for
     large scale acoustic modeling." INTERSPEECH, 2014.

    The class uses optional peep-hole connections, optional cell clipping, and
    an optional projection layer.
    """

    def __init__(self, num_units,
                 use_peepholes=False, cell_clip=None,
                 initializer=None, num_proj=None, proj_clip=None,
                 num_unit_shards=None, num_proj_shards=None,
                 forget_bias=1.0, state_is_tuple=True,
                 activation=None, reuse=None):
        """Initialize the parameters for an LSTM cell.

        Args:
          num_units: int, The number of units in the LSTM cell
          use_peepholes: bool, set True to enable diagonal/peephole connections.
          cell_clip: (optional) A float value, if provided the cell state is clipped
            by this value prior to the cell output activation.
          initializer: (optional) The initializer to use for the weight and
            projection matrices.
          num_proj: (optional) int, The output dimensionality for the projection
            matrices.  If None, no projection is performed.
          proj_clip: (optional) A float value.  If `num_proj > 0` and `proj_clip` is
            provided, then the projected values are clipped elementwise to within
            `[-proj_clip, proj_clip]`.
          num_unit_shards: Deprecated, will be removed by Jan. 2017.
            Use a variable_scope partitioner instead.
          num_proj_shards: Deprecated, will be removed by Jan. 2017.
            Use a variable_scope partitioner instead.
          forget_bias: Biases of the forget gate are initialized by default to 1
            in order to reduce the scale of forgetting at the beginning of
            the training. Must set it manually to `0.0` when restoring from
            CudnnLSTM trained checkpoints.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  This latter behavior will soon be deprecated.
          activation: Activation function of the inner states.  Default: `tanh`.
          reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.  If not `True`, and the existing scope already has
            the given variables, an error is raised.

          When restoring from CudnnLSTM-trained checkpoints, must use
          CudnnCompatibleLSTMCell instead.
        """
        super(LSTMCell, self).__init__(_reuse=reuse)
        if not state_is_tuple:
            logging.warn("%s: Using a concatenated state is slower and will soon be "
                         "deprecated.  Use state_is_tuple=True.", self)
        if num_unit_shards is not None or num_proj_shards is not None:
            logging.warn(
                "%s: The num_unit_shards and proj_unit_shards parameters are "
                "deprecated and will be removed in Jan 2017.  "
                "Use a variable scope with a partitioner instead.", self)

        self._num_units = num_units
        self._use_peepholes = use_peepholes
        self._cell_clip = cell_clip
        self._initializer = initializer
        self._num_proj = num_proj
        self._proj_clip = proj_clip
        self._num_unit_shards = num_unit_shards
        self._num_proj_shards = num_proj_shards
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or math_ops.tanh

        if num_proj:
            self._state_size = (
                LSTMStateTuple(num_units, num_proj)
                if state_is_tuple else num_units + num_proj)
            self._output_size = num_proj
        else:
            self._state_size = (
                LSTMStateTuple(num_units, num_units)
                if state_is_tuple else 2 * num_units)
            self._output_size = num_units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def call(self, inputs, state):
        """Run one step of LSTM.

        Args:
          inputs: input Tensor, 2D, batch x num_units.
          state: if `state_is_tuple` is False, this must be a state Tensor,
            `2-D, batch x state_size`.  If `state_is_tuple` is True, this must be a
            tuple of state Tensors, both `2-D`, with column sizes `c_state` and
            `m_state`.

        Returns:
          A tuple containing:

          - A `2-D, [batch x output_dim]`, Tensor representing the output of the
            LSTM after reading `inputs` when previous state was `state`.
            Here output_dim is:
               num_proj if num_proj was set,
               num_units otherwise.
          - Tensor(s) representing the new state of LSTM after reading `inputs` when
            the previous state was `state`.  Same type and shape(s) as `state`.

        Raises:
          ValueError: If input size cannot be inferred from inputs via
            static shape inference.
        """
        num_proj = self._num_units if self._num_proj is None else self._num_proj
        sigmoid = math_ops.sigmoid

        if self._state_is_tuple:
            (c_prev, m_prev) = state
        else:
            c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
            m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])

        dtype = inputs.dtype
        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
        scope = vs.get_variable_scope()
        with vs.variable_scope(scope, initializer=self._initializer) as unit_scope:
            if self._num_unit_shards is not None:
                unit_scope.set_partitioner(
                    partitioned_variables.fixed_size_partitioner(
                        self._num_unit_shards))
            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            lstm_matrix = _linear([inputs, m_prev], 4 * self._num_units, bias=True)
            i, j, f, o = array_ops.split(
                value=lstm_matrix, num_or_size_splits=4, axis=1)
            # Diagonal connections
            if self._use_peepholes:
                with vs.variable_scope(unit_scope) as projection_scope:
                    if self._num_unit_shards is not None:
                        projection_scope.set_partitioner(None)
                    w_f_diag = vs.get_variable(
                        "w_f_diag", shape=[self._num_units], dtype=dtype)
                    w_i_diag = vs.get_variable(
                        "w_i_diag", shape=[self._num_units], dtype=dtype)
                    w_o_diag = vs.get_variable(
                        "w_o_diag", shape=[self._num_units], dtype=dtype)

            if self._use_peepholes:
                c = (sigmoid(f + self._forget_bias + w_f_diag * c_prev) * c_prev +
                     sigmoid(i + w_i_diag * c_prev) * self._activation(j))
            else:
                c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) *
                     self._activation(j))

            if self._cell_clip is not None:
                # pylint: disable=invalid-unary-operand-type
                c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
                # pylint: enable=invalid-unary-operand-type
            if self._use_peepholes:
                m = sigmoid(o + w_o_diag * c) * self._activation(c)
            else:
                m = sigmoid(o) * self._activation(c)

            if self._num_proj is not None:
                with vs.variable_scope("projection") as proj_scope:
                    if self._num_proj_shards is not None:
                        proj_scope.set_partitioner(
                            partitioned_variables.fixed_size_partitioner(
                                self._num_proj_shards))
                    m = _linear(m, self._num_proj, bias=False)

                if self._proj_clip is not None:
                    # pylint: disable=invalid-unary-operand-type
                    m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
                    # pylint: enable=invalid-unary-operand-type

        new_state = (LSTMStateTuple(c, m) if self._state_is_tuple else
                     array_ops.concat([c, m], 1))
        return m, new_state


def _enumerated_map_structure(map_fn, *args, **kwargs):
    ix = [0]

    def enumerated_fn(*inner_args, **inner_kwargs):
        r = map_fn(ix[0], *inner_args, **inner_kwargs)
        ix[0] += 1
        return r

    return nest.map_structure(enumerated_fn, *args, **kwargs)


class DropoutWrapper(RNNCell):
    """Operator adding dropout to inputs and outputs of the given cell."""

    def __init__(self, cell, input_keep_prob=1.0, output_keep_prob=1.0,
                 state_keep_prob=1.0, variational_recurrent=False,
                 input_size=None, dtype=None, seed=None):
        """Create a cell with added input, state, and/or output dropout.

        If `variational_recurrent` is set to `True` (**NOT** the default behavior),
        then the same dropout mask is applied at every step, as described in:

        Y. Gal, Z Ghahramani.  "A Theoretically Grounded Application of Dropout in
        Recurrent Neural Networks".  https://arxiv.org/abs/1512.05287

        Otherwise a different dropout mask is applied at every time step.

        Args:
          cell: an RNNCell, a projection to output_size is added to it.
          input_keep_prob: unit Tensor or float between 0 and 1, input keep
            probability; if it is constant and 1, no input dropout will be added.
          output_keep_prob: unit Tensor or float between 0 and 1, output keep
            probability; if it is constant and 1, no output dropout will be added.
          state_keep_prob: unit Tensor or float between 0 and 1, output keep
            probability; if it is constant and 1, no output dropout will be added.
            State dropout is performed on the *output* states of the cell.
          variational_recurrent: Python bool.  If `True`, then the same
            dropout pattern is applied across all time steps per run call.
            If this parameter is set, `input_size` **must** be provided.
          input_size: (optional) (possibly nested tuple of) `TensorShape` objects
            containing the depth(s) of the input tensors expected to be passed in to
            the `DropoutWrapper`.  Required and used **iff**
             `variational_recurrent = True` and `input_keep_prob < 1`.
          dtype: (optional) The `dtype` of the input, state, and output tensors.
            Required and used **iff** `variational_recurrent = True`.
          seed: (optional) integer, the randomness seed.

        Raises:
          TypeError: if cell is not an RNNCell.
          ValueError: if any of the keep_probs are not between 0 and 1.
        """
        if not _like_rnncell(cell):
            raise TypeError("The parameter cell is not a RNNCell.")
        with ops.name_scope("DropoutWrapperInit"):
            def tensor_and_const_value(v):
                tensor_value = ops.convert_to_tensor(v)
                const_value = tensor_util.constant_value(tensor_value)
                return (tensor_value, const_value)

            for prob, attr in [(input_keep_prob, "input_keep_prob"),
                               (state_keep_prob, "state_keep_prob"),
                               (output_keep_prob, "output_keep_prob")]:
                tensor_prob, const_prob = tensor_and_const_value(prob)
                if const_prob is not None:
                    if const_prob < 0 or const_prob > 1:
                        raise ValueError("Parameter %s must be between 0 and 1: %d"
                                         % (attr, const_prob))
                    setattr(self, "_%s" % attr, float(const_prob))
                else:
                    setattr(self, "_%s" % attr, tensor_prob)

        # Set cell, variational_recurrent, seed before running the code below
        self._cell = cell
        self._variational_recurrent = variational_recurrent
        self._seed = seed

        self._recurrent_input_noise = None
        self._recurrent_state_noise = None
        self._recurrent_output_noise = None

        if variational_recurrent:
            if dtype is None:
                raise ValueError(
                    "When variational_recurrent=True, dtype must be provided")

            def convert_to_batch_shape(s):
                # Prepend a 1 for the batch dimension; for recurrent
                # variational dropout we use the same dropout mask for all
                # batch elements.
                return array_ops.concat(
                    ([1], tensor_shape.TensorShape(s).as_list()), 0)

            def batch_noise(s, inner_seed):
                shape = convert_to_batch_shape(s)
                return random_ops.random_uniform(shape, seed=inner_seed, dtype=dtype)

            if (not isinstance(self._input_keep_prob, numbers.Real) or
                    self._input_keep_prob < 1.0):
                if input_size is None:
                    raise ValueError(
                        "When variational_recurrent=True and input_keep_prob < 1.0 or "
                        "is unknown, input_size must be provided")
                self._recurrent_input_noise = _enumerated_map_structure(
                    lambda i, s: batch_noise(s, inner_seed=self._gen_seed("input", i)),
                    input_size)
            self._recurrent_state_noise = _enumerated_map_structure(
                lambda i, s: batch_noise(s, inner_seed=self._gen_seed("state", i)),
                cell.state_size)
            self._recurrent_output_noise = _enumerated_map_structure(
                lambda i, s: batch_noise(s, inner_seed=self._gen_seed("output", i)),
                cell.output_size)

    def _gen_seed(self, salt_prefix, index):
        if self._seed is None:
            return None
        salt = "%s_%d" % (salt_prefix, index)
        string = (str(self._seed) + salt).encode("utf-8")
        return int(hashlib.md5(string).hexdigest()[:8], 16) & 0x7FFFFFFF

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def zero_state(self, batch_size, dtype):
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            return self._cell.zero_state(batch_size, dtype)

    def _variational_recurrent_dropout_value(
            self, index, value, noise, keep_prob):
        """Performs dropout given the pre-calculated noise tensor."""
        # uniform [keep_prob, 1.0 + keep_prob)
        random_tensor = keep_prob + noise

        # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
        binary_tensor = math_ops.floor(random_tensor)
        ret = math_ops.div(value, keep_prob) * binary_tensor
        ret.set_shape(value.get_shape())
        return ret

    def _dropout(self, values, salt_prefix, recurrent_noise, keep_prob):
        """Decides whether to perform standard dropout or recurrent dropout."""
        if not self._variational_recurrent:
            def dropout(i, v):
                return nn_ops.dropout(
                    v, keep_prob=keep_prob, seed=self._gen_seed(salt_prefix, i))

            return _enumerated_map_structure(dropout, values)
        else:
            def dropout(i, v, n):
                return self._variational_recurrent_dropout_value(i, v, n, keep_prob)

            return _enumerated_map_structure(dropout, values, recurrent_noise)

    def __call__(self, inputs, state, scope=None):
        """Run the cell with the declared dropouts."""

        def _should_dropout(p):
            return (not isinstance(p, float)) or p < 1

        if _should_dropout(self._input_keep_prob):
            inputs = self._dropout(inputs, "input",
                                   self._recurrent_input_noise,
                                   self._input_keep_prob)
        output, new_state = self._cell(inputs, state, scope)
        if _should_dropout(self._state_keep_prob):
            new_state = self._dropout(new_state, "state",
                                      self._recurrent_state_noise,
                                      self._state_keep_prob)
        if _should_dropout(self._output_keep_prob):
            output = self._dropout(output, "output",
                                   self._recurrent_output_noise,
                                   self._output_keep_prob)
        return output, new_state


class ResidualWrapper(RNNCell):
    """RNNCell wrapper that ensures cell inputs are added to the outputs."""

    def __init__(self, cell):
        """Constructs a `ResidualWrapper` for `cell`.

        Args:
          cell: An instance of `RNNCell`.
        """
        self._cell = cell

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def zero_state(self, batch_size, dtype):
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            return self._cell.zero_state(batch_size, dtype)

    def __call__(self, inputs, state, scope=None):
        """Run the cell and add its inputs to its outputs.

        Args:
          inputs: cell inputs.
          state: cell state.
          scope: optional cell scope.

        Returns:
          Tuple of cell outputs and new state.

        Raises:
          TypeError: If cell inputs and outputs have different structure (type).
          ValueError: If cell inputs and outputs have different structure (value).
        """
        outputs, new_state = self._cell(inputs, state, scope=scope)
        nest.assert_same_structure(inputs, outputs)

        # Ensure shapes match
        def assert_shape_match(inp, out):
            inp.get_shape().assert_is_compatible_with(out.get_shape())

        nest.map_structure(assert_shape_match, inputs, outputs)
        res_outputs = nest.map_structure(
            lambda inp, out: inp + out, inputs, outputs)
        return (res_outputs, new_state)


class DeviceWrapper(RNNCell):
    """Operator that ensures an RNNCell runs on a particular device."""

    def __init__(self, cell, device):
        """Construct a `DeviceWrapper` for `cell` with device `device`.

        Ensures the wrapped `cell` is called with `tf.device(device)`.

        Args:
          cell: An instance of `RNNCell`.
          device: A device string or function, for passing to `tf.device`.
        """
        self._cell = cell
        self._device = device

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def zero_state(self, batch_size, dtype):
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            with ops.device(self._device):
                return self._cell.zero_state(batch_size, dtype)

    def __call__(self, inputs, state, scope=None):
        """Run the cell on specified device."""
        with ops.device(self._device):
            return self._cell(inputs, state, scope=scope)


class MultiRNNCell(RNNCell):
    """RNN cell composed sequentially of multiple simple cells."""

    def __init__(self, cells, state_is_tuple=True):
        """Create a RNN cell composed sequentially of a number of RNNCells.

        Args:
          cells: list of RNNCells that will be composed in this order.
          state_is_tuple: If True, accepted and returned states are n-tuples, where
            `n = len(cells)`.  If False, the states are all
            concatenated along the column axis.  This latter behavior will soon be
            deprecated.

        Raises:
          ValueError: if cells is empty (not allowed), or at least one of the cells
            returns a state tuple but the flag `state_is_tuple` is `False`.
        """
        super(MultiRNNCell, self).__init__()
        if not cells:
            raise ValueError("Must specify at least one cell for MultiRNNCell.")
        if not nest.is_sequence(cells):
            raise TypeError(
                "cells must be a list or tuple, but saw: %s." % cells)

        self._cells = cells
        self._state_is_tuple = state_is_tuple
        if not state_is_tuple:
            if any(nest.is_sequence(c.state_size) for c in self._cells):
                raise ValueError("Some cells return tuples of states, but the flag "
                                 "state_is_tuple is not set.  State sizes are: %s"
                                 % str([c.state_size for c in self._cells]))

    @property
    def state_size(self):
        if self._state_is_tuple:
            return tuple(cell.state_size for cell in self._cells)
        else:
            return sum([cell.state_size for cell in self._cells])

    @property
    def output_size(self):
        return self._cells[-1].output_size

    def zero_state(self, batch_size, dtype):
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            if self._state_is_tuple:
                return tuple(cell.zero_state(batch_size, dtype) for cell in self._cells)
            else:
                # We know here that state_size of each cell is not a tuple and
                # presumably does not contain TensorArrays or anything else fancy
                return super(MultiRNNCell, self).zero_state(batch_size, dtype)

    def call(self, inputs, state):
        """Run this multi-layer cell on inputs, starting from state."""
        cur_state_pos = 0
        cur_inp = inputs
        new_states = []
        for i, cell in enumerate(self._cells):
            with vs.variable_scope("cell_%d" % i):
                if self._state_is_tuple:
                    if not nest.is_sequence(state):
                        raise ValueError(
                            "Expected state to be a tuple of length %d, but received: %s" %
                            (len(self.state_size), state))
                    cur_state = state[i]
                else:
                    cur_state = array_ops.slice(state, [0, cur_state_pos],
                                                [-1, cell.state_size])
                    cur_state_pos += cell.state_size
                cur_inp, new_state = cell(cur_inp, cur_state)
                new_states.append(new_state)

        new_states = (tuple(new_states) if self._state_is_tuple else
                      array_ops.concat(new_states, 1))

        return cur_inp, new_states


class _SlimRNNCell(RNNCell):
    """A simple wrapper for slim.rnn_cells."""

    def __init__(self, cell_fn):
        """Create a SlimRNNCell from a cell_fn.

        Args:
          cell_fn: a function which takes (inputs, state, scope) and produces the
            outputs and the new_state. Additionally when called with inputs=None and
            state=None it should return (initial_outputs, initial_state).

        Raises:
          TypeError: if cell_fn is not callable
          ValueError: if cell_fn cannot produce a valid initial state.
        """
        if not callable(cell_fn):
            raise TypeError("cell_fn %s needs to be callable", cell_fn)
        self._cell_fn = cell_fn
        self._cell_name = cell_fn.func.__name__
        init_output, init_state = self._cell_fn(None, None)
        output_shape = init_output.get_shape()
        state_shape = init_state.get_shape()
        self._output_size = output_shape.with_rank(2)[1].value
        self._state_size = state_shape.with_rank(2)[1].value
        if self._output_size is None:
            raise ValueError("Initial output created by %s has invalid shape %s" %
                             (self._cell_name, output_shape))
        if self._state_size is None:
            raise ValueError("Initial state created by %s has invalid shape %s" %
                             (self._cell_name, state_shape))

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state, scope=None):
        scope = scope or self._cell_name
        output, state = self._cell_fn(inputs, state, scope=scope)
        return output, state


def _linear(args,
            output_size,
            bias,
            bias_initializer=None,
            kernel_initializer=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_initializer: starting value to initialize the bias
        (default is all zeros).
      kernel_initializer: starting value to initialize the weight.

    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
        if shape.ndims != 2:
            raise ValueError("linear is expecting 2D arguments: %s" % shapes)
        if shape[1].value is None:
            raise ValueError("linear expects shape[1] to be provided for shape %s, "
                             "but saw %s" % (shape, shape[1]))
        else:
            total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    scope = vs.get_variable_scope()
    with vs.variable_scope(scope) as outer_scope:
        weights = vs.get_variable(
            _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],
            dtype=dtype,
            initializer=kernel_initializer)
        if len(args) == 1:
            res = math_ops.matmul(args[0], weights)
        else:
            res = math_ops.matmul(array_ops.concat(args, 1), weights)
        if not bias:
            return res
        with vs.variable_scope(outer_scope) as inner_scope:
            inner_scope.set_partitioner(None)
            if bias_initializer is None:
                bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
            biases = vs.get_variable(
                _BIAS_VARIABLE_NAME, [output_size],
                dtype=dtype,
                initializer=bias_initializer)
        return nn_ops.bias_add(res, biases)
