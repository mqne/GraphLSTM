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
# todo: change the copyright notice and license
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


# identifiers for node attributes
_CELL = "cell"
_CONFIDENCE = "confidence"
_INDEX = "index"

# templates for which weights should be shared between cells
ALL_LOCAL = []
ALL_GLOBAL = [
    "W_u",
    "W_f",
    "W_c",
    "W_o",
    "U_u",
    "U_f",
    "U_c",
    "U_o",
    "U_un",
    "U_fn",
    "U_cn",
    "U_on",
    "b_u",
    "b_f",
    "b_c",
    "b_o"]
NEIGHBOUR_CONNECTIONS_GLOBAL = [
    "U_un",
    "U_fn",
    "U_cn",
    "U_on"]


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
                 state_is_tuple=True, activation=None, reuse=None, name=None):
        """Initialize the Graph LSTM cell.

        Args:
          num_units: int, The number of units in the Graph LSTM cell.
          forget_bias: float, The bias added to forget gates (see above).
            Must set to `0.0` manually when restoring from CudnnLSTM-trained
            checkpoints. This parameter is currently ignored.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  The latter behavior will soon be deprecated.
          activation: Activation function of the inner states.  Default: `tanh`.
            This parameter is currently ignored.
          reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.  If not `True`, and the existing scope already has
            the given variables, an error is raised.

          When restoring from CudnnLSTM-trained checkpoints, must use
          CudnnCompatibleLSTMCell instead.
        """
        super(GraphLSTMCell, self).__init__(_reuse=reuse, name=name)
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

    def __call__(self, inputs, state, neighbour_states, net_scope, *args, **kwargs):
        """Store neighbour_states as cell variable and call superclass.

        `__call__` is the function called by tensorflow's `dynamic_rnn`.
        It stores `neighbour_states` in a cell variable and relays the rest
        to the `__call__` method of the superclass, which in the end will call
        GraphLSTMNet's `call` method.

        Args:
          inputs: `2-D` tensor with shape `[batch_size x input_size]`.
          state: An `LSTMStateTuple` of state tensors, each shaped
            `[batch_size x self.state_size]`, if `state_is_tuple` has been set to
            `True`.  Otherwise, a `Tensor` shaped
            `[batch_size x 2 * self.state_size]`.
          neighbour_states: a list of n `LSTMStateTuples` of state tensors (m_j, h_j)
          *args: additional positional arguments to be passed to `self.call`.
          **kwargs: additional keyword arguments to be passed to `self.call`.
            **Note**: kwarg `scope` is reserved for use by the layer.

        Returns:
          A tuple, containing the new hidden state and the new state (either a
            `LSTMStateTuple` or a concatenated state, depending on
            `state_is_tuple`).
        """
        self._neighbour_states = neighbour_states
        self._net_scope = net_scope
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
        # TODO: maybe implement hand-specific version

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
        g_fij = [sigmoid(_graphlstm_linear([w_f, u_fn, b_f], [inputs, h_j], self.output_size, True,
                                           reuse_weights=None if j == 0 else [w_f, u_fn, b_f]))
                 for j, h_j in enumerate(h_j_all)]
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
    def create_nxgraph(list_or_nxgraph, num_units=None, confidence_dict=None, is_sorted=False, verify=True,
                       ignore_cell_type=False, allow_selfloops=False, **graphlstmcell_kwargs):
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
                try:
                    nxgraph.nodes[node_name][_CONFIDENCE] = float(confidence)
                except KeyError as e:
                    raise KeyError("Node '%s' in confidence_dict does not exist in nxgraph." % node_name) from e
        for node_name, node_dict in nxgraph.nodes(data=True):
            if _CONFIDENCE not in node_dict:
                nxgraph.nodes[node_name][_CONFIDENCE] = 0.
        # register index
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
                nxgraph.nodes[node_name][_CELL] = GraphLSTMCell(num_units, name="GraphLSTMCell_" + str(node_name),
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
                if not ignore_cell_type:  # todo: verify same output size for all cells
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

    @staticmethod
    def transpose_output_from_cells_first_to_batch_first(output, time_major=False):
        """Transpose a GraphLSTMNet output to the tf.nn.dynamic_rnn input format.

        GraphLSTMNet accepts input in the shape [batch_size, number_of_nodes, inputs_size],
        but the output is shaped [number_of_nodes, batch_size, output_size].
        This method transposes the output to the input format, including the time dimension
        introduced by tf.nn.dynamic_rnn.

        Args:
          output: A tensor of dimension [number_of_nodes, batch_size, output_size] or
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

    def __init__(self, nxgraph, num_units=None, state_is_tuple=True, shared_weights=ALL_GLOBAL,
                 weight_initializer=None, bias_initializer=None, name=None):
        """Create a Graph LSTM Network composed of a graph of GraphLSTMCells.

        Args:
          nxgraph: A networkx.Graph OR something a networkx.Graph can be built from.
          num_units (int): Required if building the nxgraph inside the GraphLSTMNet.
          state_is_tuple: If True, accepted and returned states are n-tuples, where
            `n = len(cells)`.  If False, the states are all
            concatenated along the column axis.  This latter behavior will soon be
            deprecated.
          shared_weights: A list of the weights that will be shared between all cells.
            Default: ALL_GLOBAL.

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
        self._weight_initializer = weight_initializer
        self._bias_initializer = bias_initializer
        if not state_is_tuple:
            if any(nest.is_sequence(self._cell(n).state_size) for n in self._nxgraph):
                raise ValueError("Some cells return tuples of states, but the flag "
                                 "state_is_tuple is not set.  State sizes are: %s"
                                 % str([self._cell(n).state_size for n in self._nxgraph]))

        # TODO init weights (W*, U*, U*n, b*) here for global weights

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

        # weights shared between cells (e.g. Ufn) and weights unique for each cell (e.g. Uf): how to handle?
        # ^this is for global Ufn local Un, which is NOT in the original paper! Paper: everything is global
        # TODO: all weights global, tf.AUTO_REUSE in cell? init all weights in net?
        # TODO: initialize global variables here in network, hand dict of global variables to cell.
        # cell initializes local variables

        # initialize variables shared between all cells
        with vs.variable_scope("shared_weights"):
            for i, w in enumerate(self._shared_weights):
                weight = vs.get_variable(
                    name=w, shape=[num_units, num_units],
                    dtype=inputs.dtype,
                    initializer=self._weight_initializer)

        new_states = [None] * self._nxgraph.number_of_nodes()
        graph_output = [None] * self._nxgraph.number_of_nodes()

        # iterate over cells in graph, starting with highest confidence value
        for it, (node_name, node_obj) in enumerate(sorted(self._nxgraph.nodes(data=True),
                                                          key=lambda x: x[1][_CONFIDENCE], reverse=True)):
            # # set scope according to if weights are being shared between all cells
            # if self._shared_weights:
            #     cell_scope = "global_weights"
            # else:
            #     cell_scope = "node_%s" % node_name
            with vs.variable_scope("node_%s" % node_name, reuse=True if self._shared_weights and it > 0 else None):  # TODO: variable scope here? in other places?
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
                cur_output, new_state = cell(cur_inp, cur_state, neighbour_states, vs.get_variable_scope())
                # store cell output and state in graph vector
                graph_output[i] = cur_output
                new_states[i] = new_state

        # pack results and return
        graph_output = tuple(graph_output)
        new_states = (tuple(new_states) if self._state_is_tuple else
                      array_ops.concat(new_states, 1))

        return graph_output, new_states


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
