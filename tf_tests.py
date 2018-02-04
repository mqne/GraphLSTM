import tensorflow as tf
import rnn_cell_impl as rci
import unittest

def main(*argv):

    sess = tf.Session()

    x = tf.constant([[1., 2.], [3., 4.]])
    y = tf.constant([[5., 6.], [7., 8.]])
    z = tf.stack((x, y))

    a = tf.unstack(z, axis=1)

    b = tf.reduce_mean(z, axis=[0])
    #x += y
    #x += y
    #print sess.run(x)

    #with tf.variable_scope("scope_tests_m") as outer_scope:
    #    z = rci._linear([x,y],4,False)
    #    c = tf.get_variable("c", [1], initializer=tf.random_normal_initializer())

    # y = tf.variables_initializer([z])
    #tf.initialize_all_variables()
    #tf.scope

    gl = rci._graphlstm_linear
    _ = gl("ll", x, 20, bias=False, weight_initializer=tf.constant_initializer([[0, 1], [-1, 1]]))

    l = rci._linear(x, 20, False)
    m = rci._linear(y, 10, False)

    scope = "test_scope"
    with tf.variable_scope(scope) as outer_scope:
        w = tf.convert_to_tensor([[0, 1], [1, 0]], name="w")
        x = tf.convert_to_tensor([[3, 2]], name="x")
        with tf.variable_scope(outer_scope, reuse=tf.AUTO_REUSE):

            result = gl("ll", x, 20, bias=False) # todo: understand
            result2 = gl("ll", x, 20, bias=False)
    print result
    print x
    print x.get_shape()[1]

    sess.run(tf.global_variables_initializer())
    print sess.run({'result': result})


class LSM(unittest.TestCase):
    def setUp(self):
        self.longMessage = True

    def test_abc(self):
        self.assertTrue(0, msg="oh shit, %s ain't true")

    def test_cdf(self):
        self.assertEqual(0,1)
        self.assertEqual(0,2)


main()
