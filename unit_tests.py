import graph_lstm as glstm
import networkx as nx
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn_cell_impl as orig_rci
import graph as rci_graph
import unittest
import matplotlib.pyplot as plt

# test graph: 20 nodes
_kickoff_hand = [("t0", "wrist"), ("i0", "wrist"), ("m0", "wrist"), ("r0", "wrist"), ("p0", "wrist"), ("i0", "m0"),
                 ("m0", "r0"), ("r0", "p0"), ("t0", "t1"), ("t1", "t2"), ("i0", "i1"), ("i1", "i2"), ("i2", "i3"),
                 ("m0", "m1"), ("m1", "m2"), ("m2", "m3"), ("r0", "r1"), ("r1", "r2"), ("r2", "r3"), ("p0", "p1"),
                 ("p1", "p2"), ("p2", "p3")]

_CELL = glstm._CELL
_INDEX = glstm._INDEX
_CONFIDENCE = glstm._CONFIDENCE


# cell that always returns fixed value on call()
class DummyFixedCell(orig_rci.RNNCell):

    def __init__(self, return_value=None, state_is_tuple=True):
        super(DummyFixedCell, self).__init__()
        self._returnValue = return_value
        self._state_is_tuple = state_is_tuple

    @property
    def state_size(self):
        return None

    @property
    def output_size(self):
        return None

    def call(self, inputs, state, neighbour_states):
        return self._returnValue


class DummyFixedTfCell(orig_rci.RNNCell):
    def __init__(self, num_units=1, memory_state=((2.,),), hidden_state=((3.,),), state_is_tuple=True):
        if not state_is_tuple:
            raise NotImplementedError("DummyFixedTfCell is only defined for state_is_tuple=True")
        super(DummyFixedTfCell, self).__init__()
        self._num_units = num_units
        self._m = tf.constant(memory_state)
        self._h = tf.constant(hidden_state)

    @property
    def state_size(self):
        return self._num_units, self._num_units

    @property
    def output_size(self):
        return self._num_units

    # get neighbour_states from net without embedding it in the state itself
    def __call__(self, inputs, state, neighbour_states, *args, **kwargs):
        self._neighbour_states = neighbour_states
        return super(DummyFixedTfCell, self).__call__(inputs, state, *args, **kwargs)

    def call(self, inputs, state):
        scope = tf.get_variable_scope()
        with tf.variable_scope(scope) as outer_scope:
            return self._h, (self._m, self._h)


# cell that always returns inputs, state, and neighbour_states on call()
class DummyReturnCell(orig_rci.RNNCell):

    def __init__(self, state_is_tuple=True):
        super(DummyReturnCell, self).__init__()
        self._state_is_tuple = state_is_tuple

    @property
    def state_size(self):
        return None

    @property
    def output_size(self):
        return None

    def call(self, inputs, state, neighbour_states):
        return (inputs, state, neighbour_states), (neighbour_states, state, inputs)


# cell that always returns (-inputs) and state or sum of neighbour states
class DummyReturnTfCell(orig_rci.RNNCell):
    def __init__(self, num_units, state_is_tuple=True, return_sum_of_neighbour_states=False,
                 add_one_to_state_per_timestep=False):
        if not state_is_tuple:
            raise NotImplementedError("DummyReturnTfCell is only defined for state_is_tuple=True")
        super(DummyReturnTfCell, self).__init__()
        self._num_units = num_units
        self._return_sum_of_neighbour_states = return_sum_of_neighbour_states
        self._add_one = add_one_to_state_per_timestep

    @property
    def state_size(self):
        return self._num_units, self._num_units

    @property
    def output_size(self):
        return self._num_units

    # get neighbour_states from net without embedding them in the state itself
    def __call__(self, inputs, state, neighbour_states, *args, **kwargs):
        self._neighbour_states = neighbour_states
        return super(DummyReturnTfCell, self).__call__(inputs, state, *args, **kwargs)

    def call(self, inputs, state):
        if self._return_sum_of_neighbour_states:
            state = tf.add_n([m for m, h in self._neighbour_states]), tf.add_n([h for m, h in self._neighbour_states])
        elif self._add_one:
            state = tuple(x + 1 for x in state)
        return -inputs, state


# cell that always returns (-inputs) and state or sum of neighbour states
# and inherits non-unique stuff from GraphLSTMCell
class DummyReturnTfGLSTMCell(glstm.GraphLSTMCell):
    def __init__(self, num_units, state_is_tuple=True, return_sum_of_neighbour_states=False,
                 add_one_to_state_per_timestep=False, *args, **kwargs):
        if not state_is_tuple:
            raise NotImplementedError("DummyReturnTfGLSTMCell is only defined for state_is_tuple=True")
        super(DummyReturnTfGLSTMCell, self).__init__(num_units, state_is_tuple=state_is_tuple, *args, **kwargs)
        self._return_sum_of_neighbour_states = return_sum_of_neighbour_states
        self._add_one = add_one_to_state_per_timestep

    # testing state_size, output_size and __call__ by not overriding them

    def call(self, inputs, state):
        # same as DummyReturnTfCell.call(self, inputs, state) with addition of own state each timestep
        if self._return_sum_of_neighbour_states:
            state = glstm.LSTMStateTuple(tf.add_n([m for m, h in self._neighbour_states]) + state[0],
                                         tf.add_n([h for m, h in self._neighbour_states]) + state[1])
        elif self._add_one:
            state = glstm.LSTMStateTuple(*[x + 1 for x in state])
        return -inputs, state


# feeds its GraphLSTMCell the same neighbour_states vector each timestep
class DummyNeighbourHelperNet(orig_rci.RNNCell):
    def __init__(self, cell, neighbour_states):
        super(DummyNeighbourHelperNet, self).__init__()
        assert isinstance(cell, glstm.GraphLSTMCell)
        self._cell = cell
        # neighbour_states dimensions: num_neighbours, batch_size, num_units
        self._neighbour_states = neighbour_states

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)

    def call(self, inputs, state):
        return self._cell(inputs, state, self._neighbour_states)


# for overriding _graphlstm_linear in graph_lstm.py, returns vector of 'value' of expected output size
class DummyGraphlstmLinear:
    def __init__(self, value):
        self._value = value

    def __call__(self, weight_names, args, output_size, bias, **kwargs):
        from tensorflow.python.util import nest
        if args is None or (nest.is_sequence(args) and not args):
            raise ValueError("`args` must be specified")
        if not nest.is_sequence(args):
            args = [args]
        dtype = [a.dtype for a in args][0]
        shape = [args[0].get_shape()[0].value, output_size]

        return tf.zeros(shape=shape, dtype=dtype) + self._value

    def get_expected_state(self, t, batch_size, num_units, neighbours_m_values):
        m = self._m_it_recursive(t, batch_size, num_units, neighbours_m_values)
        h = np.tanh(sigmoid(self._value) * m)
        return orig_rci.LSTMStateTuple(m, h)

    def _m_it_recursive(self, t, batch_size, num_units, neighbours_m_values):
        if t <= 0:
            return np.zeros([batch_size, num_units])
        return sigmoid(self._value) * (
                np.average(neighbours_m_values, axis=0)
                + self._m_it_recursive(t - 1, batch_size, num_units, neighbours_m_values)
                + np.tanh(self._value))


# this is basically a numpy implementation of a GraphLSTM cell with fixed weights for cross-verification
def get_expected_state_full(cell_input, t, batch_size, num_units, neighbours_m_values, neighbours_h_values):

        if t <= 0:
            return orig_rci.LSTMStateTuple(np.zeros([batch_size, num_units]), np.zeros([batch_size, num_units]))

        # these values depend on the initializers in TestGraphLSTMCell.test_call_full
        w_u = 1
        w_f = 2
        w_c = 3
        w_o = 4
        u_u = 5
        u_f = 6
        u_c = 7
        u_o = 8
        u_un = 9
        u_fn = 10
        u_cn = 11
        u_on = 12
        b_u = -12
        b_f = -11
        b_c = -10
        b_o = -9

        h_avg = np.average(neighbours_h_values, axis=0)

        m_old, h_old = get_expected_state_full(cell_input, t - 1, batch_size, num_units,
                                               neighbours_m_values, neighbours_h_values)

        g_f = sigmoid(cell_input[:, t - 1] * w_f + h_old * u_f + b_f)
        g_u = sigmoid(cell_input[:, t - 1] * w_u + h_old * u_u + h_avg * u_un + b_u)
        g_o = sigmoid(cell_input[:, t - 1] * w_o + h_old * u_o + h_avg * u_on + b_o)
        g_c = np.tanh(cell_input[:, t - 1] * w_c + h_old * u_c + h_avg * u_cn + b_c)

        m = np.average([sigmoid(cell_input[:, t - 1] * w_f + h_j * u_fn + b_f) * m_j
                        for m_j, h_j in zip(neighbours_m_values, neighbours_h_values)], axis=0) \
            + g_f * m_old + g_u * g_c
        h = np.tanh(g_o * m)

        return orig_rci.LSTMStateTuple(m, h)


class TestGraphLSTMNet(tf.test.TestCase):

    def setUp(self):
        self.longMessage = True
        self.G = nx.Graph(_kickoff_hand)

    def test_create_nxgraph(self):
        # template for creating graph a-b-c
        v_template = [("c", "b"), ("b", "a")]
        # the method to be tested
        cg = glstm.GraphLSTMNet.create_nxgraph

        # invalid graphs
        self.assertRaises(ValueError, cg, None)
        self.assertRaises(TypeError, cg, "Teststring")
        self.assertRaises(TypeError, cg, 5)
        self.assertRaises(ValueError, cg, [])
        self.assertRaises(TypeError, cg, ["1", "2", "2"])

        # valid graph, but invalid keywords
        # num_units < 1
        self.assertRaises(ValueError, cg, v_template, 0)
        # num_units must be int
        self.assertRaises(TypeError, cg, v_template, 1.0)
        self.assertRaises(TypeError, cg, v_template, "one")
        # confidence_dict must be a dict
        self.assertRaises(TypeError, cg, v_template, 1, confidence_dict=False)
        # confidence_dict may not contain invalid node names
        self.assertRaises(KeyError, cg, v_template, 1, confidence_dict={"z": 1})
        # confidence_dict may not contain invalid confidence values
        self.assertRaises(ValueError, cg, v_template, 1, confidence_dict={"a": "A"})

        # valid graph, valid keywords
        v_graph = cg(v_template, 6, confidence_dict={"a": .3, "c": -500})
        self.assertIs(v_graph.number_of_nodes(), 3)
        self.assertIs(v_graph.number_of_edges(), 2)
        self.assertEqual(sorted(v_graph.nodes()), sorted(['a', 'b', 'c']))
        for n in ['a', 'b', 'c']:
            self.assertEqual(v_graph.node[n].keys(), {_CONFIDENCE, _INDEX, _CELL})
        self.assertEqual(v_graph.node['a'][_CONFIDENCE], .3)
        self.assertEqual(v_graph.node['b'][_CONFIDENCE], 0)
        self.assertEqual(v_graph.node['c'][_CONFIDENCE], -500)
        self.assertEqual(v_graph.node['a'][_INDEX], 0)
        self.assertEqual(v_graph.node['b'][_INDEX], 1)
        self.assertEqual(v_graph.node['c'][_INDEX], 2)
        for n in ['a', 'b', 'c']:
            self.assertIsInstance(v_graph.node[n][_CELL], glstm.GraphLSTMCell)
            self.assertEqual(v_graph.node[n][_CELL].output_size, 6)
            self.assertEqual(v_graph.node[n][_CELL].name, "GraphLSTMCell_" + n)

        # **kwargs
        # invalid keyword
        self.assertRaises(TypeError, cg, v_template, 6, invalid_keyword=99)
        # valid keywords
        v_graph = cg(v_template, 6, forget_bias=99, activation="xyz")
        for n in ['a', 'b', 'c']:
            self.assertEqual(v_graph.node[n][_CELL]._forget_bias, 99)
            self.assertEqual(v_graph.node[n][_CELL]._activation, "xyz")

    def test_init(self):
        # GraphLSTMNet should complain when initiated with something else than a nx.Graph
        # like an int ...
        self.assertRaises(TypeError, glstm.GraphLSTMNet, 3)
        # ... None ...
        self.assertRaises(ValueError, glstm.GraphLSTMNet, None)
        # ... or nothing at all
        self.assertRaises(TypeError, glstm.GraphLSTMNet)

    def test__cell(self):
        test_node = "test_node"
        gnet = glstm.GraphLSTMNet(self.G, num_units=1, name="unittest_setup_gnet")
        gnet._nxgraph.add_node(test_node)
        # GraphLSTMNet._cell should complain when asked for non-existent node ...
        self.assertRaises(KeyError, gnet._cell, "_")
        # ... or existing node without a cell
        self.assertRaises(KeyError, gnet._cell, test_node)
        # Check if return values for existing cells are right
        gnet._nxgraph.node[test_node][_CELL] = 123
        self.assertEqual(gnet._cell(test_node), 123)
        glcell = glstm.GraphLSTMCell
        b = glcell(1)
        gnet._nxgraph.node["t0"][_CELL] = b
        self.assertIs(gnet._cell("t0"), b)
        b = glcell(1)
        self.assertIsNot(gnet._cell("t0"), b)
        self.assertIsInstance(gnet._cell("wrist"), glstm.GraphLSTMCell)

    @unittest.skip("'inputs' is a tensor when called by tensorflow. Threw no errors as of 2018-02-27,"
                   "maybe implement with tensor-input later")
    def test_call_uninodal_notf(self):
        cell_input, cell_state_m, cell_state_h, cell_cur_output, cell_new_state = objects(5)

        unet, cname = self.get_uninodal_graphlstmnet()

        # test correct returning of cell return value
        unet._nxgraph.node[cname][_CELL] = DummyFixedCell((cell_cur_output, cell_new_state)).call
        net_output = unet.call(([cell_input]), ((cell_state_m, cell_state_h),))
        expected = ((cell_cur_output,), (cell_new_state,))
        self.assertEqual(net_output, expected, msg="GraphLSTNet.call() did not return expected objects. "
                                                   "There is probably an error in GraphLSTMNet AFTER calling the cell.")

        # test correct delivering of parameters to cell
        unet._nxgraph.node[cname][_CELL] = DummyReturnCell().call
        net_output = unet.call(([cell_input]), ((cell_state_m, cell_state_h),))
        expected = (((cell_input, (cell_state_m, cell_state_h), tuple()),),
                    ((tuple(), (cell_state_m, cell_state_h), cell_input),))
        self.assertEqual(net_output, expected, msg="GraphLSTNet.call() did not deliver expected objects to cell. "
                                                   "There is probably an error in GraphLSTMNet BEFORE calling the cell.")

        # check proper index handling: uninodal GraphLSTM should complain about indices > 0
        unet._nxgraph.node[cname][_INDEX] = 1
        self.assertRaises(IndexError, unet.call, ([cell_input]), ((cell_state_m, cell_state_h),))

    def test_call_uninodal_tf(self):
        # set up net
        net, cell_name = self.get_uninodal_graphlstmnet()

        # init cells
        # constant return value 1-dim
        constant_cell_1 = DummyFixedTfCell()
        # m = [[1,2],[3,4],[5,6]], h=[[7,8],[9,10],[11,12]] (batch size 3, state size 2)
        constant_cell_2 = DummyFixedTfCell(num_units=2, memory_state=((1., 2.), (3., 4.), (5., 6.)),
                                           hidden_state=((7., 8.), (9., 10.), (11., 12.)))
        # simple return cell, 4 units
        return_cell_1 = DummyReturnTfCell(4)
        # 3 units, increase state by 1 each time step
        return_cell_2 = DummyReturnTfCell(3, add_one_to_state_per_timestep=True)

        # dimensions: batch_size, max_time, [cell dimensions] e.g. for
        #   GraphLSTMCell: input_size
        #   GraphLSTMNet: number_of_nodes, input_size

        # fixed cell 1: 1 unit, input values arbitrary

        # shape check test: too many dimensions
        input_data_xc1_e1 = tf.placeholder(tf.float32, [None, None, 2, 1, 4])
        # shape check test: wrong number of graph nodes
        input_data_xc1_e2 = tf.placeholder(tf.float32, [None, None, 2, 4])

        # input size 4: [1 1 1 4]
        input_data_xc1 = tf.placeholder(tf.float32, [None, None, 1, 4])
        feed_dict_xc1a = {input_data_xc1: [[[[6, 5, 4, 3]]]]}
        # input_size is ignored by constant cell
        cc1a_expected_result = [[[[3]]]], (([[2]], [[3]]),)

        # 1000 timesteps: [1 1000 1 1]
        input_data_cc1b = tf.placeholder(tf.float32, [None, None, 1, 1])
        feed_dict_cc1b = {input_data_cc1b: np.random.rand(1, 1000, 1, 1)}
        # timesteps are managed by dynamic_rnn
        cc1b_expected_result = [[[[3]] * 1000]], (([[2]], [[3]]),)

        # batch size 3: [3 1 1 1]
        feed_dict_cc1c = {input_data_cc1b: [[[[4]]], [[[17]]], [[[-9]]]]}
        # batch_size is ignored by constant cell
        cc1c_expected_result = [[[[3]]]], (([[2]], [[3]]),)

        # fixed cell 2: 3 units, input values arbitrary

        # batch size 3, 4 timesteps, input size 5
        input_data_cc2 = tf.placeholder(tf.float32, [None, None, 1, 5])
        feed_dict_cc2 = {
            input_data_cc2: np.random.rand(3, 4, 1, 5)}
        cc2_expected_result = [[[[7, 8]] * 4, [[9, 10]] * 4, [[11, 12]] * 4]], \
                              (([[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]),)

        # return cell 1: 4 units

        # input size 4: [1 1 1 4] (unmodified state -> zero-state)
        rc1a_expected_result = [[[[-6, -5, -4, -3]]]], (([[0, 0, 0, 0]], [[0, 0, 0, 0]]),)

        # 1000 timesteps, input size 4: [1 1000 1 4]
        rc1b_input_values = np.random.rand(1, 1000, 1, 4)
        feed_dict_rc1b = {input_data_xc1: rc1b_input_values}
        rc1b_expected_result = [-np.squeeze(rc1b_input_values, 2)], (([[0, 0, 0, 0]], [[0, 0, 0, 0]]),)

        # batch size 3, input size 4: [3 1 1 4]
        rc1c_input_values = np.random.rand(3, 1, 1, 4)
        feed_dict_rc1c = {input_data_xc1: rc1c_input_values}
        rc1c_expected_result = [-np.squeeze(rc1c_input_values, 2)], (([[0, 0, 0, 0]] * 3, [[0, 0, 0, 0]] * 3),)

        # return cell 2: 3 units, add_one_to_state_per_timestep True

        # batch size 2, 5 timesteps, input size 3:
        input_data_rc2 = tf.placeholder(tf.float32, [None, None, 1, 3])
        rc2_input_values = np.random.rand(2, 5, 1, 3)
        feed_dict_rc2 = {input_data_rc2: rc2_input_values}
        rc2_expected_result = [-np.squeeze(rc2_input_values, 2)], (([[5, 5, 5]] * 2, [[5, 5, 5]] * 2),)

        with self.test_session() as sess:
            # return value of GraphLSTMNet: graph_output, new_states
            # return value of DummyFixedTfCell: output, (state, output)
            # return value of DummyReturnTfCell: input, state
            # return value of dynamic_rnn: output [number_of_nodes, batch_size, max_time, cell.output_size],
            #   [number_of_nodes, final_state]

            # if tests containing DummyFixedTfCells fail, this might mean there are problems in GraphLSTMNet
            # AFTER calling the cell
            msg = "Calling GraphLSTNet with dummy cells did not return expected values. " \
                  "This could mean there is an error in GraphLSTMNet AFTER calling the cell."

            # inject first fixed-cell into graph
            net._nxgraph.node[cell_name][_CELL] = constant_cell_1

            self.assertRaisesRegex(ValueError, "Input shape mismatch: .* but saw 4", tf.nn.dynamic_rnn, net,
                                   input_data_xc1_e1, dtype=tf.float32)
            self.assertRaisesRegex(ValueError,
                                   "Number of nodes in GraphLSTMNet input \(2\) "
                                   "does not match number of graph nodes \(1\)",
                                   tf.nn.dynamic_rnn, net, input_data_xc1_e2, dtype=tf.float32)

            cc1a_returned_tensors = tf.nn.dynamic_rnn(net, input_data_xc1, dtype=tf.float32)
            cc1a_actual_result = sess.run(cc1a_returned_tensors, feed_dict=feed_dict_xc1a)
            np.testing.assert_equal(cc1a_actual_result, cc1a_expected_result, err_msg=msg)

            cc1bc_returned_tensors = tf.nn.dynamic_rnn(net, input_data_cc1b, dtype=tf.float32)
            cc1b_actual_result = sess.run(cc1bc_returned_tensors, feed_dict=feed_dict_cc1b)
            np.testing.assert_equal(cc1b_actual_result, cc1b_expected_result, err_msg=msg)

            cc1c_actual_result = sess.run(cc1bc_returned_tensors, feed_dict=feed_dict_cc1c)
            np.testing.assert_equal(cc1c_actual_result, cc1c_expected_result, err_msg=msg)

            # inject second fixed-cell into graph
            net._nxgraph.node[cell_name][_CELL] = constant_cell_2

            cc2_returned_tensors = tf.nn.dynamic_rnn(net, input_data_cc2, dtype=tf.float32)
            cc2_actual_result = sess.run(cc2_returned_tensors, feed_dict=feed_dict_cc2)
            np.testing.assert_equal(cc2_actual_result, cc2_expected_result, err_msg=msg)

            # if tests containing DummyReturnTfCells fail, while those containing DummyFixedTfCells
            # do not, this might mean there are problems in GraphLSTMNet BEFORE calling the cell
            msg = "Calling GraphLSTNet with return cells did not return expected values. " \
                  "This could mean there is an error in GraphLSTMNet BEFORE calling the cell."

            # inject first return-cell into graph
            net._nxgraph.node[cell_name][_CELL] = return_cell_1

            rc1_returned_tensor = tf.nn.dynamic_rnn(net, input_data_xc1, dtype=tf.float32)
            rc1a_actual_result = sess.run(rc1_returned_tensor, feed_dict=feed_dict_xc1a)
            np.testing.assert_equal(rc1a_actual_result, rc1a_expected_result, err_msg=msg)

            rc1b_actual_result = sess.run(rc1_returned_tensor, feed_dict=feed_dict_rc1b)
            np.testing.assert_allclose(rc1b_actual_result[0], rc1b_expected_result[0], err_msg=msg)
            np.testing.assert_equal(rc1b_actual_result[1], rc1b_expected_result[1], err_msg=msg)

            rc1c_actual_result = sess.run(rc1_returned_tensor, feed_dict=feed_dict_rc1c)
            np.testing.assert_allclose(rc1c_actual_result[0], rc1c_expected_result[0], err_msg=msg)
            np.testing.assert_equal(rc1c_actual_result[1], rc1c_expected_result[1], err_msg=msg)

            # inject second return-cell into graph
            net._nxgraph.node[cell_name][_CELL] = return_cell_2

            rc2_returned_tensor = tf.nn.dynamic_rnn(net, input_data_rc2, dtype=tf.float32)
            rc2_actual_result = sess.run(rc2_returned_tensor, feed_dict=feed_dict_rc2)
            np.testing.assert_allclose(rc2_actual_result[0], rc2_expected_result[0], err_msg=msg)
            np.testing.assert_equal(rc2_actual_result[1], rc2_expected_result[1], err_msg=msg)

    @staticmethod
    def get_uninodal_graphlstmnet(cell_name="node0", confidence=0):
        graph = nx.Graph()
        graph.add_node(cell_name)
        nxgraph = glstm.GraphLSTMNet.create_nxgraph(graph, 1, confidence_dict={cell_name: confidence})
        net = glstm.GraphLSTMNet(nxgraph)
        return net, cell_name

    def test_call_multinodal_tf(self):
        # graph:
        #
        #     +---c
        # a---b   |
        #     +---d

        # populate network with constant cells
        nxgraph = glstm.GraphLSTMNet.create_nxgraph([['a', 'b'], ['b', 'c'], ['b', 'd'], ['c', 'd']], 1,
                                                    confidence_dict={'a': np.random.rand(), 'b': np.random.rand(),
                                                                     'c': np.random.rand(), 'd': np.random.rand(), })
        constant_cell_a = DummyFixedTfCell(1, memory_state=((0.,),), hidden_state=((1.,),))
        constant_cell_b = DummyFixedTfCell(1, memory_state=((2.,),), hidden_state=((3.,),))
        constant_cell_c = DummyFixedTfCell(1, memory_state=((4.,),), hidden_state=((5.,),))
        constant_cell_d = DummyFixedTfCell(1, memory_state=((6.,),), hidden_state=((7.,),))

        # create GraphLSTMNet for later replacement of nodes and evaluation
        net = glstm.GraphLSTMNet([[1, 2]], 1)
        net._nxgraph = nxgraph

        # input dimensions: batch_size, max_time, [cell dimensions] e.g. for
        #   GraphLSTMCell: input_size
        #   GraphLSTMNet: number_of_nodes, input_size

        # return value of GraphLSTMNet: graph_output, new_states
        # return value of DummyFixedTfCell: output, (state, output)
        # return value of DummyReturnTfCell: input, state
        # return value of dynamic_rnn: output [number_of_nodes, batch_size, max_time, cell.output_size],
        #   final_state [number_of_nodes, state_size (2 for LSTM), batch_size, output_size]

        # fixed cells: 4 cells, 1 unit, input values arbitrary

        # input size 2: [1 1 4 2]
        input_data_cc = tf.placeholder(tf.float32, [None, None, 4, 2])
        feed_dict_cc = {input_data_cc: [[[[314, 159], [265, 358], [979, 323], [846, 264]]]]}
        # input_size is ignored by constant cell
        cc_expected_result = ([[[1]]], [[[3]]], [[[5]]], [[[7]]]), \
                             (([[0]], [[1]]), ([[2]], [[3]]), ([[4]], [[5]]), ([[6]], [[7]]))

        # make return cells
        return_cell_a = DummyReturnTfCell(3)
        return_cell_b = DummyReturnTfCell(3)
        return_cell_c = DummyReturnTfCell(3)
        return_cell_d = DummyReturnTfCell(3)

        # input size 3, batch size 5: [5 1 4 3]
        input_data_rc = tf.placeholder(tf.float32, [5, 1, 4, 3])
        rc_values = np.random.rand(5, 1, 4, 3)
        feed_dict_rc = {input_data_rc: rc_values}
        # state shape: number_of_nodes 4, state_size 2, batch_size 5, output_size 3
        rc_expected_result = -np.swapaxes(np.swapaxes(rc_values, 0, 2), 1, 2), np.zeros([4, 2, 5, 3])

        # make return cells with inter-neighbour communication
        # c increases its state by 1 each timestep, starting from 0
        # a, b and d return the sum of their neighbouring states
        return_neighbour_cell_a = DummyReturnTfCell(2, return_sum_of_neighbour_states=True)
        return_neighbour_cell_b = DummyReturnTfCell(2, return_sum_of_neighbour_states=True)
        return_neighbour_cell_c = DummyReturnTfCell(2, add_one_to_state_per_timestep=True)
        return_neighbour_cell_d = DummyReturnTfCell(2, return_sum_of_neighbour_states=True)

        # update order: c, d, a, b
        confidence_dict_cdab = {"c": 1, "d": 0.9, "a": .6, "b": -2}
        # input size 2, 4 nodes: [? ? 4 2]
        input_data_rcn_cdab = tf.placeholder(tf.float32, [None, None, 4, 2])

        # time sequence of state values:
        #   t   a   b   c   d
        #   1   0   2   1   1
        #   2   2   8   2   4
        #   3   8   22  3   11
        #   4   22  52  4   26

        # batch size 5, 1 timesteps, number_of_nodes 4, input/output size 2: [5 1 4 2]
        rcn_cdab_t1_values = np.random.rand(5, 1, 4, 2)
        feed_dict_rcn_cdab_t1 = {input_data_rcn_cdab: rcn_cdab_t1_values}
        # state shape: number_of_nodes 4, state_size 2, batch_size 5, output_size 3
        rcn_cdab_t1_expected_output = -np.swapaxes(np.swapaxes(rcn_cdab_t1_values, 0, 2), 1, 2)
        rcn_cdab_t1_expected_final_state = (
            np.zeros([2, 5, 2]), np.zeros([2, 5, 2]) + 2, np.ones([2, 5, 2]), np.ones([2, 5, 2]))

        # batch size 2, 4 timesteps, number_of_nodes 4, input/output size 2: [2 4 4 2]
        rcn_cdab_t4_values = np.random.rand(2, 4, 4, 2)
        feed_dict_rcn_cdab_t4 = {input_data_rcn_cdab: rcn_cdab_t4_values}
        # state shape: number_of_nodes 4, state_size 2, batch_size 2, output_size 3
        rcn_cdab_t4_expected_output = -np.swapaxes(np.swapaxes(rcn_cdab_t4_values, 0, 2), 1, 2)
        rcn_cdab_t4_expected_final_state = (
            np.zeros([2, 2, 2]) + 22, np.zeros([2, 2, 2]) + 52, np.zeros([2, 2, 2]) + 4, np.zeros([2, 2, 2]) + 26)

        with self.test_session() as sess:
            # if tests containing DummyFixedTfCells fail, this might mean there are problems in GraphLSTMNet
            # AFTER calling the cell
            msg = "Calling GraphLSTNet with dummy cells did not return expected values. " \
                  "This could mean there is an error in GraphLSTMNet AFTER calling the cell."

            # inject fixed cells into network graph
            nxgraph.node['a'][_CELL] = constant_cell_a
            nxgraph.node['b'][_CELL] = constant_cell_b
            nxgraph.node['c'][_CELL] = constant_cell_c
            nxgraph.node['d'][_CELL] = constant_cell_d

            cc_returned_tensors = tf.nn.dynamic_rnn(net, input_data_cc, dtype=tf.float32)

            self.assertEqual(len(cc_returned_tensors[0]), len(nxgraph),
                             msg="GraphLSTMNet should return %i outputs (equaling number of nodes), but returned %i"
                                 % (len(nxgraph), len(cc_returned_tensors[0])))

            cc_actual_result = sess.run(cc_returned_tensors, feed_dict=feed_dict_cc)

            np.testing.assert_equal(cc_actual_result, cc_expected_result, err_msg=msg)

            # if tests containing DummyReturnTfCells fail, while those containing DummyFixedTfCells
            # do not, this might mean there are problems in GraphLSTMNet BEFORE calling the cell
            msg = "Calling GraphLSTNet with return cells did not return expected values. " \
                  "This could mean there is an error in GraphLSTMNet BEFORE calling the cell."

            # inject return cells into network graph
            nxgraph.node['a'][_CELL] = return_cell_a
            nxgraph.node['b'][_CELL] = return_cell_b
            nxgraph.node['c'][_CELL] = return_cell_c
            nxgraph.node['d'][_CELL] = return_cell_d

            rc_returned_tensors = tf.nn.dynamic_rnn(net, input_data_rc, dtype=tf.float32)

            self.assertEqual(len(rc_returned_tensors[0]), len(nxgraph),
                             msg="GraphLSTMNet should return %i outputs (equaling number of nodes), but returned %i"
                                 % (len(nxgraph), len(rc_returned_tensors[0])))

            rc_actual_result = sess.run(rc_returned_tensors, feed_dict=feed_dict_rc)

            np.testing.assert_allclose(rc_actual_result[0], rc_expected_result[0], err_msg=msg)
            np.testing.assert_allclose(rc_actual_result[1], rc_expected_result[1], err_msg=msg)

            # inject neighbour-aware return cells into network graph
            nxgraph.node['a'][_CELL] = return_neighbour_cell_a
            nxgraph.node['b'][_CELL] = return_neighbour_cell_b
            nxgraph.node['c'][_CELL] = return_neighbour_cell_c
            nxgraph.node['d'][_CELL] = return_neighbour_cell_d

            # inject confidence dict for update order c, d, a, b
            net._nxgraph = glstm.GraphLSTMNet.create_nxgraph(nxgraph, confidence_dict=confidence_dict_cdab,
                                                             ignore_cell_type=True)

            rcn_cdab_returned_tensors = tf.nn.dynamic_rnn(net, input_data_rcn_cdab, dtype=tf.float32)

            self.assertEqual(len(rcn_cdab_returned_tensors[0]), len(nxgraph),
                             msg="GraphLSTMNet should return %i outputs (equaling number of nodes), but returned %i"
                                 % (len(nxgraph), len(rcn_cdab_returned_tensors[0])))

            # test batch size 5, 1 timestep
            rcn_cdab_actual_result = sess.run(rcn_cdab_returned_tensors, feed_dict=feed_dict_rcn_cdab_t1)

            np.testing.assert_allclose(rcn_cdab_actual_result[0], rcn_cdab_t1_expected_output, err_msg=msg)
            np.testing.assert_allclose(rcn_cdab_actual_result[1], rcn_cdab_t1_expected_final_state, err_msg=msg)

            # test batch size 2, 4 timesteps
            rcn_cdab_actual_result = sess.run(rcn_cdab_returned_tensors, feed_dict=feed_dict_rcn_cdab_t4)

            np.testing.assert_allclose(rcn_cdab_actual_result[0], rcn_cdab_t4_expected_output, err_msg=msg)
            np.testing.assert_allclose(rcn_cdab_actual_result[1], rcn_cdab_t4_expected_final_state, err_msg=msg)


class TestGraphLSTMCell(tf.test.TestCase):

    def setUp(self):
        self.longMessage = True
        # store original _graphlstm_linear method for restoring glstm import after test
        from graph_lstm import _graphlstm_linear as original_graphlstm_linear
        self._original_graphlstm_linear = original_graphlstm_linear

    def tearDown(self):
        # restore original _graphlstm_linear method
        glstm._graphlstm_linear = self._original_graphlstm_linear

    def test_init_and___call__(self):
        cell_name = "testcell"
        num_units = 3
        batch_size = 2
        time_steps = 4

        dummy_glstm_cell = DummyReturnTfGLSTMCell(num_units, return_sum_of_neighbour_states=True, name=cell_name)

        self.assertEqual(dummy_glstm_cell.name, cell_name)
        self.assertEqual(dummy_glstm_cell.output_size, num_units)
        self.assertIsInstance(dummy_glstm_cell.state_size, glstm.LSTMStateTuple)
        self.assertEqual(dummy_glstm_cell.state_size, (num_units, num_units))

        # input dimensions: batch_size, max_time, [cell dimensions] e.g. for
        #   GraphLSTMCell: input_size
        #   GraphLSTMNet: number_of_nodes, input_size

        # return value of GraphLSTMNet: graph_output, new_states
        # return value of DummyFixedTfCell: output, (state, output)
        # return value of DummyReturnTfCell: input, state
        # return value of dynamic_rnn: output [number_of_nodes, batch_size, max_time, cell.output_size],
        #   final_state [number_of_nodes, state_size (2 for LSTM), batch_size, output_size]

        cell_inputs_values = np.random.rand(batch_size, time_steps, dummy_glstm_cell.output_size)
        cell_inputs = tf.placeholder(tf.float32, cell_inputs_values.shape)

        neighbour_state_1_values_c = np.random.rand(batch_size, num_units)
        neighbour_state_1_values_h = np.random.rand(batch_size, num_units)
        neighbour_state_2_values_c = np.random.rand(batch_size, num_units)
        neighbour_state_2_values_h = np.random.rand(batch_size, num_units)

        # not explicitly converting with dtype=tf.float32 destroys the whole testsuite in weird ways,
        # inflicting errors in parts not at all connected to this one

        state_neighbour_1_t4 = glstm.LSTMStateTuple(tf.convert_to_tensor(neighbour_state_1_values_c, dtype=tf.float32),
                                                    tf.convert_to_tensor(neighbour_state_1_values_h, dtype=tf.float32))
        state_neighbour_2_t4 = glstm.LSTMStateTuple(tf.convert_to_tensor(neighbour_state_2_values_c, dtype=tf.float32),
                                                    tf.convert_to_tensor(neighbour_state_2_values_h, dtype=tf.float32))

        cell_neighbour_states_t4 = (state_neighbour_1_t4, state_neighbour_2_t4)

        expected_output = -cell_inputs_values
        # state shape: state_size 2, batch_size 2, output_size 3
        expected_final_state = np.sum([[neighbour_state_1_values_c, neighbour_state_1_values_h],
                                       [neighbour_state_2_values_c, neighbour_state_2_values_h]], axis=0) * time_steps

        helper_net = DummyNeighbourHelperNet(dummy_glstm_cell, cell_neighbour_states_t4)

        with self.test_session() as sess:
            return_tensor = tf.nn.dynamic_rnn(helper_net, cell_inputs, dtype=tf.float32)

            actual_result = sess.run(return_tensor, feed_dict={cell_inputs: cell_inputs_values})

            np.testing.assert_allclose(actual_result[0], expected_output)
            np.testing.assert_allclose(actual_result[1], expected_final_state)

    def test_call_without__graphlstm_linear(self):
        # patch _graphlstm_linear method to stub for this test
        glstm._graphlstm_linear = DummyGraphlstmLinear((np.random.randn() - .5) * 20)

        num_units = 3
        batch_size = 2
        time_steps = 4

        glstm_cell = glstm.GraphLSTMCell(num_units)

        self.assertEqual(glstm_cell.output_size, num_units)
        self.assertIsInstance(glstm_cell.state_size, glstm.LSTMStateTuple)
        self.assertEqual(glstm_cell.state_size, (num_units, num_units))

        # input dimensions: batch_size, max_time, [cell dimensions] e.g. for
        #   GraphLSTMCell: input_size
        #   GraphLSTMNet: number_of_nodes, input_size

        # return value of GraphLSTMNet: graph_output, new_states
        # return value of DummyFixedTfCell: output, (state, output)
        # return value of DummyReturnTfCell: input, state
        # return value of dynamic_rnn: output [number_of_nodes, batch_size, max_time, cell.output_size],
        #   final_state [number_of_nodes, state_size (2 for LSTM), batch_size, output_size]

        cell_inputs_values = np.random.rand(batch_size, time_steps, glstm_cell.output_size)
        cell_inputs = tf.placeholder(tf.float32, cell_inputs_values.shape)

        neighbour_state_1_values_c = np.random.rand(batch_size, num_units)
        neighbour_state_1_values_h = np.random.rand(batch_size, num_units)
        neighbour_state_2_values_c = np.random.rand(batch_size, num_units)
        neighbour_state_2_values_h = np.random.rand(batch_size, num_units)

        # not explicitly converting with dtype=tf.float32 destroys the whole testsuite in weird ways,
        # inflicting errors in parts not at all connected to this one

        state_neighbour_1_t4 = glstm.LSTMStateTuple(tf.convert_to_tensor(neighbour_state_1_values_c, dtype=tf.float32),
                                                    tf.convert_to_tensor(neighbour_state_1_values_h, dtype=tf.float32))
        state_neighbour_2_t4 = glstm.LSTMStateTuple(tf.convert_to_tensor(neighbour_state_2_values_c, dtype=tf.float32),
                                                    tf.convert_to_tensor(neighbour_state_2_values_h, dtype=tf.float32))

        cell_neighbour_states_t4 = (state_neighbour_1_t4, state_neighbour_2_t4)

        # state shape: state_size 2, batch_size 2, output_size 3
        expected_final_state = glstm._graphlstm_linear.get_expected_state(time_steps, batch_size, num_units,
                                                                          [neighbour_state_1_values_c,
                                                                           neighbour_state_2_values_c])
        expected_output = np.swapaxes([glstm._graphlstm_linear.get_expected_state(t, batch_size, num_units,
                                                                                  [neighbour_state_1_values_c,
                                                                                   neighbour_state_2_values_c])[1]
                                       for t in range(1, time_steps + 1)], 0, 1)

        helper_net = DummyNeighbourHelperNet(glstm_cell, cell_neighbour_states_t4)

        with self.test_session() as sess:
            return_tensor = tf.nn.dynamic_rnn(helper_net, cell_inputs, dtype=tf.float32)

            actual_result = sess.run(return_tensor, feed_dict={cell_inputs: cell_inputs_values})

            err_msg = "Possibly helpful for debugging: _graphlstm_linear return value was set to %f" % \
                      glstm._graphlstm_linear._value

            np.testing.assert_allclose(actual_result[0], expected_output, atol=1e-5, err_msg=err_msg)
            np.testing.assert_allclose(actual_result[1], expected_final_state, atol=1e-5, err_msg=err_msg)

    def test_call_full(self):

        num_units = 3
        batch_size = 2
        time_steps = 4

        glstm_cell = glstm.GraphLSTMCell(num_units)

        self.assertEqual(glstm_cell.output_size, num_units)
        self.assertIsInstance(glstm_cell.state_size, glstm.LSTMStateTuple)
        self.assertEqual(glstm_cell.state_size, (num_units, num_units))

        # input dimensions: batch_size, max_time, [cell dimensions] e.g. for
        #   GraphLSTMCell: input_size
        #   GraphLSTMNet: number_of_nodes, input_size

        # return value of GraphLSTMNet: graph_output, new_states
        # return value of DummyFixedTfCell: output, (state, output)
        # return value of DummyReturnTfCell: input, state
        # return value of dynamic_rnn: output [number_of_nodes, batch_size, max_time, cell.output_size],
        #   final_state [number_of_nodes, state_size (2 for LSTM), batch_size, output_size]

        cell_inputs_values = np.random.rand(batch_size, time_steps, glstm_cell.output_size)
        cell_inputs = tf.placeholder(tf.float32, cell_inputs_values.shape)

        neighbour_state_1_values_c = np.random.rand(batch_size, num_units)
        neighbour_state_1_values_h = np.random.rand(batch_size, num_units)
        neighbour_state_2_values_c = np.random.rand(batch_size, num_units)
        neighbour_state_2_values_h = np.random.rand(batch_size, num_units)

        # not explicitly converting with dtype=tf.float32 destroys the whole testsuite in weird ways,
        # inflicting errors in parts not at all connected to this one

        state_neighbour_1_t4 = glstm.LSTMStateTuple(tf.convert_to_tensor(neighbour_state_1_values_c, dtype=tf.float32),
                                                    tf.convert_to_tensor(neighbour_state_1_values_h, dtype=tf.float32))
        state_neighbour_2_t4 = glstm.LSTMStateTuple(tf.convert_to_tensor(neighbour_state_2_values_c, dtype=tf.float32),
                                                    tf.convert_to_tensor(neighbour_state_2_values_h, dtype=tf.float32))

        cell_neighbour_states_t4 = (state_neighbour_1_t4, state_neighbour_2_t4)

        # state shape: state_size 2, batch_size 2, output_size 3
        expected_final_state = get_expected_state_full(cell_inputs_values, time_steps, batch_size, num_units,
                                                       [neighbour_state_1_values_c, neighbour_state_2_values_c],
                                                       [neighbour_state_1_values_h, neighbour_state_2_values_h])
        expected_output = np.swapaxes([get_expected_state_full(cell_inputs_values, t, batch_size, num_units,
                                                               [neighbour_state_1_values_c, neighbour_state_2_values_c],
                                                               [neighbour_state_1_values_h, neighbour_state_2_values_h]
                                                               )[1]
                                       for t in range(1, time_steps + 1)], 0, 1)

        helper_net = DummyNeighbourHelperNet(glstm_cell, cell_neighbour_states_t4)

        # initialize weights to specific values
        scope = tf.get_variable_scope()
        # note: the scope name is sensitive to changes in the architecture of the test
        with tf.variable_scope("rnn/dummy_neighbour_helper_net/graph_lstm_cell") as init_scope:

            # weight and bias names (copied from GraphLSTMCell.call) # and values
            weights = [
                "W_u",    # 1
                "W_f",    # 2
                "W_c",    # 3
                "W_o",    # 4
                "U_u",    # 5
                "U_f",    # 6
                "U_c",    # 7
                "U_o",    # 8
                "U_un",   # 9
                "U_fn",  # 10
                "U_cn",  # 11
                "U_on"]  # 12
            biases = [
                "b_u",  # -12
                "b_f",  # -11
                "b_c",  # -10
                "b_o"]   # -9

            # initialize variables via tf.get_variable()
            for i, w in enumerate(weights):
                tf.get_variable(name=w, shape=[num_units, num_units], initializer=tf.initializers.identity(i+1))
            for i, b in enumerate(biases):
                tf.get_variable(name=b, shape=[num_units], initializer=tf.constant_initializer(i-12))

        # force reuse=True for all variables in order to use variables as initialized above
        with tf.variable_scope(scope, reuse=True) as reuse_scope:
            with self.test_session() as sess:
                return_tensor = tf.nn.dynamic_rnn(helper_net, cell_inputs, dtype=tf.float32)

                tf.global_variables_initializer().run()
                actual_result = sess.run(return_tensor, feed_dict={cell_inputs: cell_inputs_values})

                np.testing.assert_allclose(actual_result[0], expected_output, rtol=1e-4)
                np.testing.assert_allclose(actual_result[1], expected_final_state, rtol=1e-4)


class TestGraphLSTMCellAndNet(tf.test.TestCase):

    def setUp(self):
        self.longMessage = True

    def test_cell_init_and___call__in_full_net(self):
        # create a DummyReturnTfCell calling/inheriting init and __call__ and test it
        # basically repeat the last test from TestGraphLSTMNet.test_call_multinodal_tf
        # with a DummyReturnTfGLSTMCell

        # update order: c, d, a, b
        nxgraph = glstm.GraphLSTMNet.create_nxgraph([['a', 'b'], ['b', 'c'], ['b', 'd'], ['c', 'd']], 1,
                                                    confidence_dict={"c": 1, "d": 0.9, "a": .6, "b": -2})
        net = glstm.GraphLSTMNet(nxgraph)

        # make return cells with inter-neighbour communication
        # c increases its state by 1 each timestep, starting from 0
        # a, b and d return the sum of their neighbouring states
        return_neighbour_cell_a = DummyReturnTfGLSTMCell(2, return_sum_of_neighbour_states=True)
        return_neighbour_cell_b = DummyReturnTfGLSTMCell(2, return_sum_of_neighbour_states=True)
        return_neighbour_cell_c = DummyReturnTfGLSTMCell(2, add_one_to_state_per_timestep=True)
        return_neighbour_cell_d = DummyReturnTfGLSTMCell(2, return_sum_of_neighbour_states=True)

        # input size 2, 4 nodes: [? ? 4 2]
        input_data_rcn_cdab = tf.placeholder(tf.float32, [None, None, 4, 2])

        # time sequence of state values:
        #   t   a   b   c   d
        #   1   0   2   1   1
        #   2   2   11  2   5
        #   3   13  46  3   19
        #   4   59  178 4   69

        # batch size 2, 4 timesteps, number_of_nodes 4, input/output size 2: [2 4 4 2]
        rcn_cdab_t4_values = np.random.rand(2, 4, 4, 2)
        feed_dict_rcn_cdab_t4 = {input_data_rcn_cdab: rcn_cdab_t4_values}
        # state shape: number_of_nodes 4, state_size 2, batch_size 2, output_size 3
        rcn_cdab_t4_expected_output = -np.swapaxes(np.swapaxes(rcn_cdab_t4_values, 0, 2), 1, 2)
        rcn_cdab_t4_expected_final_state = (
            np.zeros([2, 2, 2]) + 59, np.zeros([2, 2, 2]) + 178, np.zeros([2, 2, 2]) + 4, np.zeros([2, 2, 2]) + 69)

        with self.test_session() as sess:
            # inject neighbour-aware return cells into network graph
            nxgraph.node['a'][_CELL] = return_neighbour_cell_a
            nxgraph.node['b'][_CELL] = return_neighbour_cell_b
            nxgraph.node['c'][_CELL] = return_neighbour_cell_c
            nxgraph.node['d'][_CELL] = return_neighbour_cell_d

            rcn_cdab_returned_tensors = tf.nn.dynamic_rnn(net, input_data_rcn_cdab, dtype=tf.float32)

            self.assertEqual(len(rcn_cdab_returned_tensors[0]), len(nxgraph),
                             msg="GraphLSTMNet should return %i outputs (equaling number of nodes), but returned %i"
                                 % (len(nxgraph), len(rcn_cdab_returned_tensors[0])))

            # test batch size 2, 4 timesteps
            rcn_cdab_actual_result = sess.run(rcn_cdab_returned_tensors, feed_dict=feed_dict_rcn_cdab_t4)

            np.testing.assert_allclose(rcn_cdab_actual_result[0], rcn_cdab_t4_expected_output)
            np.testing.assert_allclose(rcn_cdab_actual_result[1], rcn_cdab_t4_expected_final_state)


class TestGraphLSTMLinear(tf.test.TestCase):

    def setUp(self):
        self.longMessage = True
        self.func = glstm._graphlstm_linear

        self.x = tf.constant([[1., 2.], [3., 4.]])
        self.y = tf.constant([[5., 6.], [7., 8.]])
        self.z = tf.constant([[0., 1.], [2., 3.], [4., 5.]])
        self.custom_initializer_1 = tf.constant_initializer([[0, -1], [2, 1]])

    def test_errors(self):
        self.assertRaisesRegex(ValueError, "args", self.func, ['_'], [], 1, True)
        self.assertRaisesRegex(ValueError, "weight_names", self.func, [], ['_'], 1, True)
        self.assertRaisesRegex(ValueError, "True.*one element longer", self.func, ['1', '2'], ['1', '2'], 1, True)
        self.assertRaisesRegex(ValueError, "False.*same length", self.func, ['1', '2'], ['1'], 1, False)
        self.assertRaisesRegex(LookupError, "`reuse_weights`", self.func, ['1', '2'], ['3', '4'], 1, False,
                               reuse_weights=['3'])

    def test_calc(self):
        w1 = "weight_name_1"
        w2 = "weight_name_2"
        w3 = "weight_name_3"
        w4 = "weight_name_4"
        b1 = "bias_name_1"
        b2 = "bias_name_2"
        b3 = "bias_name_3"
        n = "non_existing_name"

        # x * w1 (1 1,1 1)
        glxw1 = self.func([w1], self.x, 2, False, weight_initializer=tf.ones_initializer)
        glxw1_expected_result = [[3, 3], [7, 7]]
        # existing variable should throw error when fetched without reuse
        self.assertRaisesRegex(ValueError, "already exists", self.func, w1, self.x, 2, False)
        # new variable should throw error when fetched with reuse
        self.assertRaisesRegex(ValueError, "does not exist", self.func, n, self.x, 2, False, reuse_weights=n)
        # x * w1 + y * w2 (0 -1,2 1) + b1 (0 0)
        glxw1yw2b1 = self.func([w1, w2, b1], [self.x, self.y], 2, True, weight_initializer=self.custom_initializer_1,
                               reuse_weights=[w1])
        glxw1yw2b1_expected_result = [[15, 4], [23, 8]]
        # y * w1 + x * w2 + b2 (1 1)
        glyw1xw2b2 = self.func([w1, w2, b2], [self.y, self.x], 2, True, bias_initializer=tf.ones_initializer,
                               reuse_weights=[w1, w2])
        glyw1xw2b2_expected_result = [[16, 13], [24, 17]]
        # non-square matrices
        # x * w3 (1 1)
        glxw3 = self.func(w3, self.x, 1, False, weight_initializer=tf.ones_initializer)
        glxw3_expected_result = [[3], [7]]
        # z * w4 (1 1 1 1,1 1 1 1,1 1 1 1) + b3 (1 1 1 1)
        glzw4b3 = self.func([w4, b3], self.z, 4, True, weight_initializer=tf.ones_initializer,
                            bias_initializer=tf.ones_initializer)
        glzw4b3_expected_result = [[2, 2, 2, 2], [6, 6, 6, 6], [10, 10, 10, 10]]

        with self.test_session():
            tf.global_variables_initializer().run()

            np.testing.assert_equal(glxw1.eval(), glxw1_expected_result)
            np.testing.assert_equal(glxw1yw2b1.eval(), glxw1yw2b1_expected_result)
            np.testing.assert_equal(glyw1xw2b2.eval(), glyw1xw2b2_expected_result)
            np.testing.assert_equal(glxw3.eval(), glxw3_expected_result)
            np.testing.assert_equal(glzw4b3.eval(), glzw4b3_expected_result)


# print node information for graph or GraphLSTMNet g
def print_node(name, g):
    if isinstance(g, glstm.GraphLSTMNet):
        print("Node information for GraphLSTMNet %s:" % str(g))
        g = g._nxgraph
    else:
        print("Node information for graph %s:" % str(g))
    print("graph[\"%s\"]: %s" % (name, str(g[name])))
    print("graph.node[\"%s\"]: %s" % (name, str(g.node[name])))


def plot_nxgraph(net):
    plt.subplot()
    nx.draw(net._nxgraph, with_labels=True)
    plt.show()


# calculate sigmoid(x). Note: this implementation is for testing purposes only and should NOT be used for critical code!
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# return tuple of n objects
def objects(n):
    r = []
    for _ in range(n):
        r.append(object())
    return tuple(r)


def dirty_tests():
    G = nx.Graph()
    G.add_node("a", cell="heyho")
    g = glstm.GraphLSTMNet.create_nxgraph(G, ignore_cell_type=True)
    glstm.GraphLSTMNet.is_valid_nxgraph(g, ignore_cell_type=True)


def main():
    dirty_tests()
    with tf.variable_scope("unittest"):
        # TestGraphLSTMNet.test_call_multinodal_tf(TestGraphLSTMNet())
        unittest.main()


main()
