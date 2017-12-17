import tensorflow as tf

s = tf.Session()

x = tf.constant([[1., 2.], [3., 4.]])
y = tf.constant([[5., 6.], [7., 8.]])
z = tf.stack((x,y))

a = tf.unstack(z, axis=1)

b = tf.reduce_mean(z, axis=[0])

print s.run([b, z])
