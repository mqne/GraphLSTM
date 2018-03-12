import tensorflow as tf
import graph_lstm as glstm
import unittest
import numpy as np
import networkx as nx
import tensorflow.python.user_ops.user_ops

class DummyFixedTfCell(glstm.RNNCell):
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

    def call(self, inputs, state, neighbour_states=None):
        #return inputs, (inputs, inputs)
        return self._h, (self._m, self._h)

class DummyReturnTfCell(glstm.RNNCell):
    def __init__(self, num_units, state_is_tuple=True, return_sum_of_neighbour_states=False):
        if not state_is_tuple:
            raise NotImplementedError("DummyFixedTfCell is only defined for state_is_tuple=True")
        super(DummyReturnTfCell, self).__init__()
        self._num_units = num_units
        self._return_sum_of_neighbour_states = return_sum_of_neighbour_states

    @property
    def state_size(self):
        return self._num_units, self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state, neighbour_states=None):
        if self._return_sum_of_neighbour_states:
            state = tf.add_n([m for m, h in neighbour_states]), tf.add_n([h for m, h in neighbour_states])
        return inputs, tuple(x+1 for x in state)


def main(*argv):

    sess = tf.Session()

    x = tf.constant([[1., 2.], [3., 4.]])
    x1 = tf.constant([[1., 2.], [3., 4.]])
    y = tf.constant([[5., 6.], [7., 8.]])
    z = tf.stack((x, y))

    a = tf.unstack(z, axis=1)

    b = tf.reduce_mean(z, axis=[0])
    #x += y
    #x += y
    #print(sess.run(x))

    #with tf.variable_scope("scope_tests_m") as outer_scope:
    #    z = rci._linear([x,y],4,False)
    #    c = tf.get_variable("c", [1], initializer=tf.random_normal_initializer())

    # y = tf.variables_initializer([z])
    #tf.initialize_all_variables()
    #tf.scope

    gl = glstm._graphlstm_linear
    _ = gl("ll", x, 20, bias=False, weight_initializer=tf.constant_initializer([[0, 1], [-1, 1]]))

    #l = rci._linear(x, 20, False)
    #m = rci._linear(y, 10, False)

    scope = "test_scope"
    #with tf.variable_scope(scope) as outer_scope:
    #    w = tf.convert_to_tensor([[0, 1], [1, 0]], name="w")
    #    x = tf.convert_to_tensor([[3, 2]], name="x")
    #    with tf.variable_scope(outer_scope, reuse=tf.AUTO_REUSE):
#
    #        result = gl("ll", x, 20, bias=False)
    #        result2 = gl("ll", x, 20, bias=False)
    #print(result)
    #print(x)
    #print(x.get_shape()[1])

    xy1 = x1 * y
    xy2 = tf.multiply(x1, y)
    xy3 = tf.matmul(x1, y)

    gr = nx.Graph()
    gr.add_node("test")
    dftcell1 = DummyFixedTfCell(num_units=1)
    drtcell1 = DummyReturnTfCell(num_units=3)

    input_data = tf.placeholder(tf.float32, [None, None, 3])

    sess.run(tf.global_variables_initializer())
    # print(sess.run({'result': result}))
    r = sess.run({"*": xy1, "multiply": xy2, "matmul": xy3})
    print(xy1)
    print(xy2)
    print(xy3)
    print(np.array_equal(r["*"], r["multiply"]))

    #print(sess.run(tf.nn.dynamic_rnn(dftcell1, input_data, initial_state=tf.ones([2,3])), feed_dict={input_data: [[[1, 5, 6]]]}))
    print(sess.run(tf.nn.dynamic_rnn(drtcell1, input_data, dtype=tf.float32), feed_dict={input_data: [[[1, 19, 3], [1, 19, 3]]]}))
    print(sess.run(tf.nn.dynamic_rnn(drtcell1, input_data, dtype=tf.float32), feed_dict={input_data: [[[1, 19, 3]]]}))

    print(glstm.GraphLSTMNet.is_valid_nxgraph(nx.Graph(), raise_errors=False))

class LSM(unittest.TestCase):
    def setUp(self):
        self.longMessage = True

    def test_abc(self):
        self.assertTrue(0, msg="oh shit, %s ain't true")

    def test_cdf(self):
        self.assertEqual(0, 1)
        self.assertEqual(0, 2)


main()
