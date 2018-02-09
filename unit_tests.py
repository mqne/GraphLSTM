import rnn_cell_impl as rci
import networkx as nx
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn_cell_impl as orig_rci
# TODO what is tensorflow.python.user_ops for?
import graph as rci_graph
import unittest

# test graph: 20 nodes
_kickoff_hand = [("t0", "wrist"), ("i0", "wrist"), ("m0", "wrist"), ("r0", "wrist"), ("p0", "wrist"), ("i0", "m0"),
                 ("m0", "r0"), ("r0", "p0"), ("t0", "t1"), ("t1", "t2"), ("i0", "i1"), ("i1", "i2"), ("i2", "i3"),
                 ("m0", "m1"), ("m1", "m2"), ("m2", "m3"), ("r0", "r1"), ("r1", "r2"), ("r2", "r3"), ("p0", "p1"),
                 ("p1", "p2"), ("p2", "p3")]

_CELL = rci._CELL
_INDEX = rci._INDEX
_CONFIDENCE = rci._CONFIDENCE


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


class TestGraphLSTMNet(unittest.TestCase):

    def setUp(self):
        self.longMessage = True
        self.G = nx.Graph(_kickoff_hand)
        self.gnet = rci.GraphLSTMNet(self.G)

    def test_init(self):
        # GraphLSTMNet should complain when initiated with something else than a nx.Graph
        # like an int ...
        self.assertRaises(TypeError, rci.GraphLSTMNet, 3)
        # ... None ...
        self.assertRaises(ValueError, rci.GraphLSTMNet, None)
        # ... or nothing at all
        self.assertRaises(TypeError, rci.GraphLSTMNet)

    def test__cell(self):
        # GraphLSTMNet._cell should complain when asked for non-existent node ...
        self.assertRaises(KeyError, self.gnet._cell, "_")
        # ... or existing node without a cell
        self.assertRaises(KeyError, self.gnet._cell, "wrist")
        # Check if return values for existing cells are right
        self.gnet._graph.node["wrist"]["cell"] = 123
        self.assertEqual(self.gnet._cell("wrist"), 123)
        glcell = rci.GraphLSTMCell
        b = glcell(1)
        self.gnet._graph.node["t0"]["cell"] = b
        self.assertIs(self.gnet._cell("t0"), b)
        b = glcell(1)
        self.assertIsNot(self.gnet._cell("t0"), b)

    def test_call_uninodal_notf(self):
        uninodal_graph = nx.Graph()
        cname = "node0"
        uninodal_graph.add_node(cname)
        unet = rci.GraphLSTMNet(uninodal_graph)
        cell_input, cell_state_m, cell_state_h, cell_cur_output, cell_new_state = objects(5)
        unet._graph.node[cname][_CONFIDENCE] = 0
        unet._graph.node[cname][_INDEX] = 0

        # test correct returning of cell return value
        unet._graph.node[cname][_CELL] = DummyFixedCell((cell_cur_output, cell_new_state)).call
        net_output = unet.call(([cell_input]), ((cell_state_m, cell_state_h),))
        expected = ((cell_cur_output,), (cell_new_state,))
        self.assertEqual(net_output, expected, msg="GraphLSTNet.call() did not return expected objects. "
                                                   "There is probably an error in GraphLSTMNet AFTER calling the cell.")

        # test correct delivering of parameters to cell
        unet._graph.node[cname][_CELL] = DummyReturnCell().call
        net_output = unet.call(([cell_input]), ((cell_state_m, cell_state_h),))
        expected = (((cell_input, (cell_state_m, cell_state_h), tuple()),),
                    ((tuple(), (cell_state_m, cell_state_h), cell_input),))
        self.assertEqual(net_output, expected, msg="GraphLSTNet.call() did not deliver expected objects to cell. "
                                                   "There is probably an error in GraphLSTMNet BEFORE calling the cell.")

        # check proper index handling: uninodal GraphLSTM should complain about indices > 0
        unet._graph.node[cname][_INDEX] = 1
        self.assertRaises(IndexError, unet.call, ([cell_input]), ((cell_state_m, cell_state_h),))

    def test_call_uninodal_tf(self):
        # TODO
        print "Note: test_call_uninodal_tf is not implemented yet."
        # sess = tf.InteractiveSession()
        # tf.initialize_all_variables()
        pass


class TestGraphLSTMLinear(unittest.TestCase):

    def setUp(self):
        self.longMessage = True
        self.func = rci._graphlstm_linear

        self.sess = tf.Session()
        self.x = tf.constant([[1., 2.], [3., 4.]])
        self.y = tf.constant([[5., 6.], [7., 8.]])
        self.z = tf.constant([[0., 1.], [2., 3.], [4., 5.]])
        self.custom_initializer_1 = tf.constant_initializer([[0, -1], [2, 1]])

    def test_errors(self):
        self.assertRaisesRegexp(ValueError, "args", self.func, ['_'], [], 1, True)
        self.assertRaisesRegexp(ValueError, "weight_names", self.func, [], ['_'], 1, True)
        self.assertRaisesRegexp(ValueError, "True.*one element longer", self.func, ['1', '2'], ['1', '2'], 1, True)
        self.assertRaisesRegexp(ValueError, "False.*same length", self.func, ['1', '2'], ['1'], 1, False)
        self.assertRaisesRegexp(LookupError, "`reuse_weights`", self.func, ['1', '2'], ['3', '4'], 1, False,
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
        self.assertRaisesRegexp(ValueError, "already exists", self.func, w1, self.x, 2, False)
        # new variable should throw error when fetched with reuse
        self.assertRaisesRegexp(ValueError, "does not exist", self.func, n, self.x, 2, False, reuse_weights=n)
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

        self.sess.run(tf.global_variables_initializer())

        np.testing.assert_equal(self.sess.run(glxw1), glxw1_expected_result)
        np.testing.assert_equal(self.sess.run(glxw1yw2b1), glxw1yw2b1_expected_result)
        np.testing.assert_equal(self.sess.run(glyw1xw2b2), glyw1xw2b2_expected_result)
        np.testing.assert_equal(self.sess.run(glxw3), glxw3_expected_result)
        np.testing.assert_equal(self.sess.run(glzw4b3), glzw4b3_expected_result)


# print node information for graph or GraphLSTMNet g
def print_node(name, g):
    if isinstance(g, rci.GraphLSTMNet):
        print "Node information for GraphLSTMNet %s:" % str(g)
        g = g._graph
    else:
        print "Node information for graph %s:" % str(g)
    print "graph[\"%s\"]: %s" % (name, str(g[name]))
    print "graph.node[\"%s\"]: %s" % (name, str(g.node[name]))


# return tuple of n objects
def objects(n):
    r = []
    for _ in xrange(n):
        r.append(object())
    return tuple(r)


def dirty_tests():
    pass


def main():
    # test_call_uninodal_GraphLSTMNet_tf()
    dirty_tests()
    unittest.main()


main()
