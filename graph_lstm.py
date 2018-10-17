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
# This file provides an extension of TensorFlow towards Graph LSTM.
# It is implemented by Matthias Kuehne and based on the 2016 CVPR paper
# 'Semantic Object Parsing with Graph LSTM' by Liang et al., available at
# https://arxiv.org/abs/1603.07063
"""Module implementing Graph LSTM.

This module provides the Graph LSTM network, as well as the cell needed therefor.
It also implements the operator needed for the cell's internal calculations.
Constructing multi-layer networks is supported by calling the network several times.
"""
import networkx as nx

from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple, RNNCell
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow import Tensor


# identifiers for node attributes
_CELL = "cell"
_CONFIDENCE = "confidence"
_INDEX = "index"

# weight names
_W_U = "W_u"
_W_F = "W_f"
_W_C = "W_c"
_W_O = "W_o"
_U_U = "U_u"
_U_F = "U_f"
_U_C = "U_c"
_U_O = "U_o"
_U_UN = "U_un"
_U_FN = "U_fn"
_U_CN = "U_cn"
_U_ON = "U_on"
# bias names
_B_U = "b_u"
_B_F = "b_f"
_B_C = "b_c"
_B_O = "b_o"

# weight groups
_WEIGHTS = {
    _W_U,
    _W_F,
    _W_C,
    _W_O}

_UEIGHTS = {
    _U_U,
    _U_F,
    _U_C,
    _U_O}

_NEIGHBOUR_UEIGHTS = {
    _U_UN,
    _U_FN,
    _U_CN,
    _U_ON}

_BIASES = {
    _B_U,
    _B_F,
    _B_C,
    _B_O}

# templates for which weights should be shared between cells
NONE_SHARED = set()
ALL_SHARED = {*_WEIGHTS, *_UEIGHTS, *_NEIGHBOUR_UEIGHTS, *_BIASES}
NEIGHBOUR_CONNECTIONS_SHARED = {*_NEIGHBOUR_UEIGHTS}


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
    """

    def __init__(self, num_units, state_is_tuple=True, bias_initializer=None, weight_initializer=None,
                 forget_bias_initializer=None, reuse=None, name=None):
        """Initialize the Graph LSTM cell.

        Args:
          num_units: int, The number of units in the Graph LSTM cell.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  The latter behavior will soon be deprecated.
          bias_initializer: The initializer that should be used for initializing
            biases. If None, _init_weights uses a uniform initializer
            in the interval [-0.1, 0.1).
          weight_initializer: The initializer that should be used for initializing
            weights. If None, _init_weights uses a uniform initializer
            in the interval [-0.1, 0.1).
          forget_bias_initializer: The initializer that should be used for initializing
            the forget gate weight b_f. If None, _init_weights uses a
            constant initializer of 1.
          reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.  If not `True`, and the existing scope already has
            the given variables, an error is raised. Using this parameter will
            result in untested behaviour. todo: obsolete?
          name: (optional) The name that will be used for this cell in the
            tensorflow namespace.

          When restoring from CudnnLSTM-trained checkpoints, must use
          CudnnCompatibleLSTMCell instead.
        """
        super(GraphLSTMCell, self).__init__(_reuse=reuse, name=name)
        if not state_is_tuple:
            logging.warn("%s: Using a concatenated state is UNTESTED for this GraphLSTM implementation. "
                         "If it works, it is likely to be slower and will soon be "
                         "deprecated.  Use state_is_tuple=True.", self)
        self._num_units = num_units
        self._state_is_tuple = state_is_tuple
        self._bias_initializer = bias_initializer
        self._weight_initializer = weight_initializer
        self._forget_bias_initializer = forget_bias_initializer

    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def _get_weight_shape(self, weight, inputs):
        """Calculate the shape of a Graph LSTM weight.

        Those variables starting with "W" are multiplied with the input,
        those with "U" with the state of either the cell or neighbouring cells,
        and biases don't depend on anything other than the output size.
        This function calculates the corresponding shape.

        Args:
          weight: The name of the weight.
          inputs: `2-D` tensor with shape `[batch_size x input_size]`,
            the input to the cell.

        Returns:
          A tuple the shape of the weight.

        Raises:
          NotImplementedError: If a non-standard weight name is encountered.
        """
        if weight in _WEIGHTS:
            return tuple([inputs.get_shape()[1], self.output_size])
        if weight in _UEIGHTS | _NEIGHBOUR_UEIGHTS:
            return tuple([self.state_size[1] if self._state_is_tuple else self.state_size//2, self.output_size])
        if weight in _BIASES:
            return tuple([self.output_size])
        raise NotImplementedError("Inferring shape for non-standard Graph LSTM cell weights is not supported")

    def _init_weights(self, inputs):
        """Initialize the weights.

        Args:
          inputs: `2-D` tensor with shape `[batch_size x input_size]`,
            the input to the cell. Needed for calculating weight shapes.

        Returns:
          A dict of weight name:tensorflow-weight pairs.
        """
        dtype = inputs.dtype
        bias_initializer = init_ops.random_uniform_initializer(-0.1, 0.1, dtype=dtype) \
            if self._bias_initializer is None else self._bias_initializer
        weight_initializer = init_ops.random_uniform_initializer(-0.1, 0.1, dtype=dtype) \
            if self._weight_initializer is None else self._weight_initializer
        forget_bias_initializer = init_ops.constant_initializer(1.0, dtype=dtype) \
            if self._forget_bias_initializer is None else self._forget_bias_initializer

        weight_dict = {}

        # initialize shared weights
        with vs.variable_scope(self._shared_scope) as scope:
            for weight_name in self._shared_weights:
                if weight_name == _B_F:
                    with vs.variable_scope(scope) as bias_scope:
                        bias_scope.set_partitioner(None)
                        weight = vs.get_variable(
                            name=weight_name, shape=self._get_weight_shape(weight_name, inputs),
                            dtype=dtype,
                            initializer=forget_bias_initializer)
                elif weight_name not in _BIASES:
                    weight = vs.get_variable(
                        name=weight_name, shape=self._get_weight_shape(weight_name, inputs),
                        dtype=dtype,
                        initializer=weight_initializer)
                else:
                    with vs.variable_scope(scope) as bias_scope:
                        bias_scope.set_partitioner(None)
                        weight = vs.get_variable(
                            name=weight_name, shape=self._get_weight_shape(weight_name, inputs),
                            dtype=dtype,
                            initializer=bias_initializer)
                weight_dict[weight_name] = weight

        # initialize local weights
        for weight_name in _WEIGHTS | _UEIGHTS | _NEIGHBOUR_UEIGHTS:
            if weight_name not in self._shared_weights:
                weight = vs.get_variable(
                    name=weight_name, shape=self._get_weight_shape(weight_name, inputs),
                    dtype=dtype,
                    initializer=weight_initializer)
                weight_dict[weight_name] = weight
        for weight_name in _BIASES:
            if weight_name not in self._shared_weights:
                if weight_name == _B_F:
                    weight = vs.get_variable(
                        name=weight_name, shape=self._get_weight_shape(weight_name, inputs),
                        dtype=dtype,
                        initializer=forget_bias_initializer)
                else:
                    weight = vs.get_variable(
                        name=weight_name, shape=self._get_weight_shape(weight_name, inputs),
                        dtype=dtype,
                        initializer=bias_initializer)
                weight_dict[weight_name] = weight

        return weight_dict

    def __call__(self, inputs, state, neighbour_states, shared_scope, shared_weights, *args, **kwargs):
        """Store neighbour_states and shared properties as cell variable and call superclass.

        `__call__` is the function called by tensorflow's `dynamic_rnn`.
        It stores `neighbour_states`, `shared_scope` and `shared_weights`
        in cell variables and relays the rest to the `__call__` method
        of the superclass, which in the end will call GraphLSTMNet's
        `call` method.

        Args:
          inputs: `2-D` tensor with shape `[batch_size x input_size]`.
          state: An `LSTMStateTuple` of state tensors, each shaped
            `[batch_size x self.state_size]`, if `state_is_tuple` has been set to
            `True`.  Otherwise, a `Tensor` shaped
            `[batch_size x 2 * self.state_size]`.
          neighbour_states: a list of n `LSTMStateTuples` of state tensors (m_j, h_j)
          shared_scope: The tensorflow scope in which the shared variables reside.
          shared_weights: The list of names of shared variables.
          *args: additional positional arguments to be passed to `self.call`.
          **kwargs: additional keyword arguments to be passed to `self.call`.
            **Note**: kwarg `scope` is reserved for use by the layer.

        Returns:
          A tuple, containing the new hidden state and the new state (either a
            `LSTMStateTuple` or a concatenated state, depending on
            `state_is_tuple`).
        """
        self._neighbour_states = neighbour_states
        self._shared_scope = shared_scope
        self._shared_weights = shared_weights
        return super(GraphLSTMCell, self).__call__(inputs, state, *args, **kwargs)

    def call(self, inputs, state):
        """Run one step of the GraphLSTM cell.

        Args:
          inputs: `2-D` tensor with shape `[batch_size x input_size]`.
          state: An `LSTMStateTuple` of state tensors, each shaped
            `[batch_size x self.state_size]`, if `state_is_tuple` has been set to
            `True`.  Otherwise, a `Tensor` shaped
            `[batch_size x 2 * self.state_size]`.

        Returns:
          A tuple, containing the new hidden state and the new state (either a
            `LSTMStateTuple` or a concatenated state, depending on
            `state_is_tuple`).
        """
        sigmoid = math_ops.sigmoid
        tanh = math_ops.tanh

        # initialize cell weights
        weight_dict = self._init_weights(inputs)

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

        # self._neighbour_states: a list of n `LSTMStateTuples` of state tensors (m_j, h_j)
        if not hasattr(self, "_neighbour_states"):
            raise LookupError("Could not find variable 'self._neighbour_states' during 'GraphLSTMCell.call'.\n"
                              "This likely means 'call' was called directly, instead of through '__call__' (which "
                              "should be the case when called from inside the tensorflow framework).")
        # extract two vectors of n ms and n hs from state vector of n (m,h) tuples
        m_j_all, h_j_all = zip(*self._neighbour_states)

        # IMPLEMENTATION DIFFERS FROM PAPER: in eq. (2) g^f_ij uses h_j,t regardless of if node j has been updated
        # already or not. Implemented here is h_j,t for non-updated nodes and h_j,t+1 for updated nodes
        # which both makes sense intuitively (most recent information)
        # and is more lightweight (no need to keep track of old states)

        # Eq. 1: averaged hidden states for neighbouring nodes h^-_{i,t}
        h_j_avg = math_ops.reduce_mean(h_j_all, axis=0)

        # fetch weights and biases
        w_u = weight_dict[_W_U]
        w_f = weight_dict[_W_F]
        w_c = weight_dict[_W_C]
        w_o = weight_dict[_W_O]
        u_u = weight_dict[_U_U]
        u_f = weight_dict[_U_F]
        u_c = weight_dict[_U_C]
        u_o = weight_dict[_U_O]
        u_un = weight_dict[_U_UN]
        u_fn = weight_dict[_U_FN]
        u_cn = weight_dict[_U_CN]
        u_on = weight_dict[_U_ON]
        b_u = weight_dict[_B_U]
        b_f = weight_dict[_B_F]
        b_c = weight_dict[_B_C]
        b_o = weight_dict[_B_O]

        # Eq. 2
        # input gate
        # g_u = sigmoid ( f_{i,t+1} * W_u + h_{i,t} * U_u + h^-_{i,t} * U_{un} + b_u )
        g_u = sigmoid(_graphlstm_linear([w_u, u_u, u_un, b_u], [inputs, h_i, h_j_avg]))
        # adaptive forget gate
        # g_fij = sigmoid ( f_{i,t+1} * W_f + h_{j,t} * U_fn + b_f ) for every neighbour j
        g_fij = [sigmoid(_graphlstm_linear([w_f, u_fn, b_f], [inputs, h_j])) for h_j in h_j_all]
        # forget gate
        # g_fi = sigmoid ( f_{i,t+1} * W_f + h_{i,t} * U_f + b_f )
        g_fi = sigmoid(_graphlstm_linear([w_f, u_f, b_f], [inputs, h_i]))
        # output gate
        # g_o = sigmoid ( f_{i,t+1} * W_o + h_{i,t} * U_o + h^-_{i,t} * U_{on} + b_o )
        g_o = sigmoid(_graphlstm_linear([w_o, u_o, u_on, b_o], [inputs, h_i, h_j_avg]))
        # memory gate
        # g_c = tanh ( f_{i,t+1} * W_c + h_{i,t} * U_c + h^-_{i,t} * U_{cn} + b_c )
        g_c = tanh(_graphlstm_linear([w_c, u_c, u_cn, b_c], [inputs, h_i, h_j_avg]))

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

    This implementation does not support dynamic confidence values.
    The confidence values for each node at network creation time
    thus determine the update order.
    If you want to set the update order manually, create the nxgraph
    first by invoking `create_nxgraph` and then pass the resulting
    graph to the GraphLSTMNet constructor.
    """

    def _cell(self, node, nxgraph=None):
        """Return the GraphLSTMCell belonging to a node.

        Args:
          node (any): The node whose GraphLSTMCell object will be returned.
          nxgraph (networkx.Graph): The graph from which the node will be extracted.
            Defaults to self._nxgraph .
        """
        if nxgraph is None:
            nxgraph = self._nxgraph
        elif not isinstance(nxgraph, nx.classes.graph.Graph):
            raise TypeError(
                "nxgraph must be a Graph of package networkx, but saw: %s." % nxgraph)

        return nxgraph.nodes[node][_CELL]

    @staticmethod
    def create_nxgraph(list_or_nxgraph, num_units=None, confidence_dict=None, index_dict=None, is_sorted=False,
                       verify=True, ignore_cell_type=False, allow_selfloops=False, **graphlstmcell_kwargs):
        """Return a GraphLSTM Network graph (composed of GraphLSTMCells).

        Args:
          list_or_nxgraph (any): A networkx.Graph OR something a networkx.Graph
            can be built from.
          num_units (int): Required if at least one cell is not defined yet (which
            is always true if list_or_nxgraph is not a networkx.Graph). Becomes the
            num_units paramter of the GraphLSTMCells.
          confidence_dict (dict): Holds the confidence values for the nodes that should
            start off with a confidence value different from 0. Optional.
            Format: {node_name_1: confidence_1, ...}
            NOTE: currently, the confidence value is only read once, at network creation
            time. Specifying a confidence_dict when creating the graph is thus the only
            way to determine the GraphLSTM update order.
          index_dict (dict): Holds the index values for all nodes if the indices should
            be custom. If not None, parameter is_sorted is ignored. Optional.
            Format: {node_name_1: index_1, ...}
          is_sorted (bool): If True, the _INDEX parameter will be assigned in the native
            order of elements as seen in list_or_nxgraph. If False, _INDEX will
            correspond to the order given by sorted() (default).
          verify (bool): If True (default), the final nxgraph will be checked for
            validity by GraphLSTMNet.is_valid_nxgraph, and a warning will be issued
            in case of failure.
          ignore_cell_type (bool): If the verification should ignore the cells' types
            (default: False).
          allow_selfloops (bool): If the verification should allow for selfloops
            in the graph (default: False).
          **graphlstmcell_kwargs: optional keyword arguments that will get passed
            to the GraphLSTMCell constructor.

        Raises:
          ValueError, TypeError, KeyError: If the nxgraph cannot be constructed.
        """
        if list_or_nxgraph is None:
            raise ValueError("Cannot create nxgraph from None.")
        try:
            nxgraph = nx.Graph(list_or_nxgraph)
        except nx.exception.NetworkXError as e:
            raise TypeError("Cannot create nxgraph from %r" % repr(list_or_nxgraph)) from e
        if len(nxgraph) < 1:
            raise ValueError("nxgraph needs at least one node.")
        # register confidence (default: 0 (float))
        if confidence_dict is not None:
            if not isinstance(confidence_dict, dict):
                raise TypeError("confidence_dict must be of type 'dict', but found '%s'." % type(confidence_dict))
            for node_name, confidence in confidence_dict.items():
                if isinstance(confidence, Tensor):
                    raise NotImplementedError("Support for dynamic confidence values is currently not implemented. "
                                              "Please use python float values instead of Tensors.")
                try:
                    nxgraph.nodes[node_name][_CONFIDENCE] = confidence if isinstance(confidence, Tensor) else float(
                        confidence)
                except KeyError as e:
                    raise KeyError("Node '%s' in confidence_dict does not exist in nxgraph." % node_name) from e
        for node_name, node_dict in nxgraph.nodes(data=True):
            if _CONFIDENCE not in node_dict:
                nxgraph.nodes[node_name][_CONFIDENCE] = 0.
        # register index
        if index_dict is not None:
            if not isinstance(index_dict, dict):
                raise TypeError("index_dict must be of type 'dict', but found '%s'." % type(index_dict))
            if len(index_dict) != len(nxgraph):
                raise ValueError("index_dict must have as many entries as nxgraph has nodes (%i), but found %i"
                                 % (len(index_dict), len(nxgraph)))
            if sorted([int(x) for x in index_dict.values()]) != list(range(len(nxgraph))):
                raise ValueError("index_dict must contain indices ranging from 0 to number of nodes - 1, but saw\n"
                                 "%r" % sorted(index_dict.values()))
            for node_name, index in index_dict.items():
                try:
                    nxgraph.nodes[node_name][_INDEX] = int(index)
                except KeyError as e:
                    raise KeyError("Node '%s' in index_dict does not exist in nxgraph." % node_name) from e
        else:
            for index, node_name in enumerate(sorted(nxgraph) if not is_sorted else nxgraph):
                nxgraph.nodes[node_name][_INDEX] = index
        # register cells
        num_units_type_checked_flag = False
        for node_name, node_dict in nxgraph.nodes(data=True):
            if _CELL not in node_dict:
                if not num_units_type_checked_flag:
                    if num_units is None:
                        raise ValueError("Must specify num_units when creating GraphLSTMCells "
                                         "via method create_nxgraph.")
                    if not isinstance(num_units, int):
                        raise TypeError("num_units must be of type 'int', but found '%s': %s"
                                        % (type(num_units), num_units))
                    if num_units < 1:
                        raise ValueError("num_units must be a positive integer, but found: %i" % num_units)
                    num_units_type_checked_flag = True
                nxgraph.nodes[node_name][_CELL] = GraphLSTMCell(num_units, name="graph_lstm_cell_" + str(node_name),
                                                                **graphlstmcell_kwargs)
        if verify and not GraphLSTMNet.is_valid_nxgraph(nxgraph, raise_errors=False, ignore_cell_type=ignore_cell_type,
                                                        allow_selfloops=allow_selfloops):
            logging.warn("Created nxgraph did not pass validity test. "
                         "For details, run GraphLSTMNet.is_valid_nxgraph explicitly.")
        return nxgraph

    @staticmethod
    def is_valid_nxgraph(nxgraph, raise_errors=True, ignore_cell_type=False, allow_selfloops=False):
        """Check if a given graph is a valid GraphLSTMNet graph.

        Args:
          nxgraph (any): The graph to be checked.
          raise_errors (bool): If True, the method raises an error as soon as
            a problem is detected.
          ignore_cell_type (bool): If True, the graph will not be considered 'bad'
            if its cells are not GraphLSTMCells.
          allow_selfloops (bool): If True, the graph will not be considered 'bad'
            if it contains selfloops.

        Returns:
          True if graph is fine, False otherwise.

        Raises:
          TypeError: If something inside the graph (or the graph itself) is of
            the wrong type.
          ValueError: If graph contains no nodes, or the _INDEX attributes
            don't form the expected well-defined list.
          LookupError: If a node misses the _CELL, _CONFIDENCE or _INDEX attribute.
        """
        try:
            if not isinstance(nxgraph, nx.classes.graph.Graph):
                raise TypeError("nxgraph is of type %s, but should be an instance of networkx.classes.graph.Graph."
                                % str(nxgraph))
            if nxgraph.number_of_nodes() < 1:
                raise ValueError("nxgraph needs at least one node.")
            if not allow_selfloops and nx.number_of_selfloops(nxgraph) != 0:
                raise ValueError("nxgraph has %i selfloops. "
                                 "If this is expected, consider running is_valid_nxgraph with allow_selfloops=True.\n"
                                 "Nodes with selfloops: %r"
                                 % (nx.number_of_selfloops(nxgraph), list(nx.nodes_with_selfloops(nxgraph))))
            node_attr_lookuperr = None
            index_list = []
            for node_name in nxgraph.nodes:
                if _CELL not in nxgraph.nodes[node_name]:
                    node_attr_lookuperr = "_CELL"
                elif _INDEX not in nxgraph.nodes[node_name]:
                    node_attr_lookuperr = "_INDEX"
                elif _CONFIDENCE not in nxgraph.nodes[node_name]:
                    node_attr_lookuperr = "_CONFIDENCE"
                if node_attr_lookuperr is not None:
                    raise KeyError("Node '%s' has no attribute %s" % (node_name, node_attr_lookuperr))
                if not ignore_cell_type:  # todo: verify same output size for all cells? does that make sense?
                    if not isinstance(nxgraph.nodes[node_name][_CELL], GraphLSTMCell):
                        raise TypeError("Cell of node '%s' is not a GraphLSTMCell. "
                                        "If this is expected, consider running is_valid_nxgraph with "
                                        "ignore_cell_type=True." % node_name)
                if not isinstance(nxgraph.nodes[node_name][_INDEX], int):
                    raise TypeError("_INDEX attribute should always be an integer, but is not for node '%s'"
                                    % node_name)
                else:
                    index_list.append(nxgraph.nodes[node_name][_INDEX])
                if not isinstance(nxgraph.nodes[node_name][_CONFIDENCE], float):
                    raise TypeError("_CONFIDENCE attribute should always be float, but is not for node '%s'"
                                    % node_name)
            if sorted(index_list) != list(range(len(index_list))):
                raise ValueError("The values of all _INDEX attributes have to form a well-sorted list, "
                                 "starting at 0 and ending at number of nodes - 1.\n"
                                 "Expected 0 ... %i, but found:\n%s"
                                 % (len(index_list) - 1, sorted(index_list)))
        except (TypeError, ValueError, KeyError):
            if raise_errors:
                raise
            return False
        else:
            return True

    def reshape_input_for_dynamic_rnn(self, input_tensor, timesteps=None):
        """Reshape a time-dimension free Tensor to input shape required
        by GraphLSTMNet, optionally adding time dimension.

        The `input_tensor` will be reshaped to [batch_size, number_of_nodes, input_size], which is the expected
        input shape of the GraphLSTM.

        If `timesteps` is specified, a time dimension of that length is added for a resulting shape of
        [batch_size, timesteps, number_of_nodes, input_size] with the input being identical for each timestep.
        This is useful for preparing a tensor without time dimension to be fed into
        tf.dynamic_rnn(GraphLSTMNet, inputs=reshaped_tensor)

        Args:
          input_tensor: The tensor to be reshaped.
          timesteps (int): (optional) The number of timesteps to be included in the tensor. Default: None.

        Returns:
          The reshaped input tensor [batch_size,[ timesteps,] number_of_nodes, input_size].
        """
        shaped_tensor = array_ops.reshape(input_tensor, shape=[-1, len(self.output_size), self.output_size[0]])
        if timesteps is not None:
            return array_ops.stack([shaped_tensor] * timesteps, axis=1)
        return shaped_tensor

    @staticmethod
    def transpose_output_from_cells_first_to_batch_first(output, time_major=False):
        """Transpose a GraphLSTMNet output to the tf.nn.dynamic_rnn input format.

        GraphLSTMNet accepts input in the shape [batch_size, number_of_nodes, inputs_size],
        but the output is shaped [number_of_nodes, batch_size, output_size].
        This method transposes the output to the input format, including the time dimension
        introduced by tf.nn.dynamic_rnn.

        Args:
          output: A tensor of dimension
            [number_of_nodes, batch_size, max_time, output_size] if time_major=False or
            [number_of_nodes, max_time, batch_size, output_size] if time_major=True.
          time_major (bool): As used in tf.nn.dynamic_rnn. Default: False.

        Returns:
          The transposed tensor [batch_size, max_time, number_of_nodes, output_size].

        Raises:
          ValueError: If the tensor shape is invalid, i.e. the dimensionality is not 4.
        """
        output_tensor = ops.convert_to_tensor(output)
        if len(output_tensor.shape) == 4:
            if not time_major:
                perm = [1, 2, 0, 3]
            else:
                perm = [2, 1, 0, 3]
        else:
            raise ValueError("output.shape needs to be of length 4 (for tf.nn.dynamic_rnn), "
                             "but was %i. output.shape is %r"
                             % (len(output.shape), output.shape))
        return array_ops.transpose(output, perm)

    def __init__(self, nxgraph, num_units=None, state_is_tuple=True, shared_weights=ALL_SHARED, name=None):
        """Create a Graph LSTM Network composed of a graph of GraphLSTMCells.

        Args:
          nxgraph: A networkx.Graph OR something a networkx.Graph can be built from.
          num_units (int): Required if building the nxgraph inside the GraphLSTMNet.
          state_is_tuple: If True, accepted and returned states are n-tuples, where
            `n = len(cells)`.  If False, the states are all
            concatenated along the column axis.  This latter behavior will soon be
            deprecated.
          shared_weights: A list of the weights that will be shared between all cells.
            Default: ALL_SHARED.
          name (string): The Tensorflow name of the Graph LSTM network. Must be given
            if more than one is used.

        Raises:
          ValueError: If nxgraph is not valid, or at least one of the cells
            returns a state tuple but the flag `state_is_tuple` is `False`.
        """
        super(GraphLSTMNet, self).__init__(name=name)
        if not nxgraph:
            raise ValueError("Must specify nxgraph for GraphLSTMNet.")
        # check if nxgraph is a valid GraphLSTM graph, create one if not
        if not self.is_valid_nxgraph(nxgraph, raise_errors=False):
            try:
                nxgraph = self.create_nxgraph(nxgraph, num_units, verify=False)
            except ValueError as e:
                if "Must specify num_units" in str(e):
                    raise ValueError("Must specify num_units when building nxgraph inside GraphLSTMNet.init.") from None
                else:
                    raise
            # now check if graph is valid, raise errors if not
            try:
                self.is_valid_nxgraph(nxgraph)
            except (TypeError, ValueError, KeyError) as e:
                raise ValueError("Invalid nxgraph specified for GraphLSTMNet.") from e

        self._nxgraph = nxgraph
        self._state_is_tuple = state_is_tuple
        self._shared_weights = shared_weights
        if not state_is_tuple:
            if any(nest.is_sequence(self._cell(n).state_size) for n in self._nxgraph):
                raise ValueError("Some cells return tuples of states, but the flag "
                                 "state_is_tuple is not set.  State sizes are: %s"
                                 % str([self._cell(n).state_size for n in self._nxgraph]))

    @property
    def state_size(self):
        if self._state_is_tuple:
            return tuple(self._cell(n).state_size for n in self._nxgraph)
        else:
            return sum([self._cell(n).state_size for n in self._nxgraph])

    @property
    def output_size(self):
        return tuple(self._cell(n).output_size for n in self._nxgraph)

    def zero_state(self, batch_size, dtype):
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            if self._state_is_tuple:
                return tuple(self._cell(n).zero_state(batch_size, dtype) for n in self._nxgraph)
            else:
                # We know here that state_size of each cell is not a tuple and
                # presumably does not contain TensorArrays or anything else fancy
                return super(GraphLSTMNet, self).zero_state(batch_size, dtype)

    def call(self, inputs, state):
        """Run this Graph LSTM on inputs, starting from state.

        Args:
          inputs: A tensor of dimensions [batch_size, number_of_nodes, inputs_size].
            The index of each node in this tensor must correspond to the node attribute 'index'.
          state: A tuple or tensor of states for each node.
        """

        # check if input dimensions match expectation
        if len(inputs.shape) != 3:
            raise ValueError("Input shape mismatch: expected tensor of 3 dimensions "
                             "(batch_size, cell_count, input_size), but saw %i: "
                             "%s" % (len(inputs.shape), inputs.shape))
        if inputs.shape[-2] != self._nxgraph.number_of_nodes():
            raise ValueError("Number of nodes in GraphLSTMNet input (%d) does not match number of graph nodes (%d)" %
                             (inputs.shape[-2], self._nxgraph.number_of_nodes()))

        new_states = [None] * self._nxgraph.number_of_nodes()
        graph_output = [None] * self._nxgraph.number_of_nodes()

        # iterate over cells in graph, starting with highest confidence value
        for it, (node_name, node_obj) in enumerate(sorted(self._nxgraph.nodes(data=True),
                                                          key=lambda x: x[1][_CONFIDENCE], reverse=True)):

            # initialize scope for weights shared between all cells
            with vs.variable_scope("shared_weights", reuse=True if it > 0 else None) as shared_scope:
                pass

            with vs.variable_scope("node_%s" % node_name):
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
                for neighbour_name in nx.all_neighbors(self._nxgraph, node_name):
                    n_i = self._nxgraph.node[neighbour_name][_INDEX]
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
                cur_inp = inputs[:, i]
                # run current cell
                cur_output, new_state = cell(cur_inp, cur_state, neighbour_states,
                                             shared_scope, self._shared_weights)
                # store cell output and state in graph vector
                graph_output[i] = cur_output
                new_states[i] = new_state

        # pack results and return
        graph_output = tuple(graph_output)
        new_states = (tuple(new_states) if self._state_is_tuple else
                      array_ops.concat(new_states, 1))

        return graph_output, new_states


# calculates terms like W * f + U * h + b
def _graphlstm_linear(weights, args):
    """Linear map: sum_i(args[i] * weights[i]) + bias, where weights[i] and bias can be multiple variables.

    Weights are multiplied with args according to the ordering inside the lists.
      The weights parameter must be ordered such that the first part (i.e. the first len(args) elements)
      constitutes only weights, and the second part (after len(args)) only biases.

    Args:
      weights: a Tensor or a list of weight and bias Tensors.
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.

    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]) + b ( + ... + b_n) .

    Raises:
      ValueError: if an argument has unspecified or wrong shape or if trying to multiply
        a bias or adding a weight.
    """
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if weights is None or (nest.is_sequence(weights) and not weights):
        raise ValueError("`weights` must be specified")
    if not nest.is_sequence(args):
        args = [args]
    if not nest.is_sequence(weights):
        weights = [weights]

    if len(args) > len(weights):
        raise ValueError("Number of args (%i) exceeds number of weights (%i)" % (len(args), len(weights)))

    # Now the computation.
    summands = []
    for x, w in zip(args, weights[:len(args)]):
        # extract weight name from full tensorflow weight name
        name = w.name.split("/")[-1].split(":")[0]
        if name in _BIASES:
            raise ValueError("Weight scheduled for multiplication with arg found in _BIASES: %s" % name)
        summands.append(math_ops.matmul(x, w))
    res = math_ops.add_n(summands)

    for b in weights[len(args):]:
        name = b.name.split("/")[-1].split(":")[0]
        if name in _WEIGHTS | _UEIGHTS | _NEIGHBOUR_UEIGHTS:
            raise ValueError("Weight scheduled for bias addition found in %s: %s" %
                             ("_WEIGHTS" if name in _WEIGHTS else
                              "_UEIGHTS" if name in _UEIGHTS else
                              "_NEIGHBOUR_WEIGHTS",
                              name))
        res = nn_ops.bias_add(res, b)

    return res
