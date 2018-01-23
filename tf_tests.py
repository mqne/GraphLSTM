import tensorflow as tf

s = tf.Session()

x = tf.constant([[1., 2.], [3., 4.]])
y = tf.constant([[5., 6.], [7., 8.]])
z = tf.stack((x,y))

a = tf.unstack(z, axis=1)

b = tf.reduce_mean(z, axis=[0])
x += y
x += y
print s.run(x)

import unittest

class LSM(unittest.TestCase):
    def setUp(self):
        self.longMessage = True
    def test_abc(self):
        self.assertTrue(0, msg="oh shit, %s ain't true")
    def test_cdf(self):
        self.assertEqual(0,1)
        self.assertEqual(0,2)

unittest.main()