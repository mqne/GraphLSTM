import tensorflow as tf
import rnn_cell_impl as rci

s = tf.Session()

x = tf.constant([[1., 2.], [3., 4.]])
y = tf.constant([[5., 6.], [7., 8.]])
z = tf.stack((x,y))

a = tf.unstack(z, axis=1)

b = tf.reduce_mean(z, axis=[0])

with tf.variable_scope("scope_tests_m") as outer_scope:
    z = rci._linear([x,y],4,False)
    c = tf.get_variable("c", [1], initializer=tf.random_normal_initializer())

# y = tf.variables_initializer([z])
tf.initialize_all_variables()
tf.scope

print s.run([z])
