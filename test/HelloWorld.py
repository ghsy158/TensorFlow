import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
a = tf.constant(1)
b = tf.constant(2)
sum = tf.add(a, b)
sess = tf.Session()
sess.run(sum)
print(sum)
print(sess.run(hello))
sess.close()