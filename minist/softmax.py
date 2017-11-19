"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md

先定义mnist 读取数据，sess定义Session()回话，
建立模型，x占位符，W权重变量，b偏置变量，y是一个数学公式，实现线性回归(有关相性回归，我现在还说不清，只有个大体了解，先不说)，这里的y得出是一个概率值。
然后定义交叉煽，是模型的出来的值和真实值比较的得出的一个误差；再就是训练步长，作用是减小这个误差，例子中就是通过梯度下降法来减小误差，这就是涉及到多次训练，怎么多次训练，就是下面说的
进行1000次反复训练，每次从mnist中批量取出100张图片，然后放入sess跑起来。到这里整个模型就跑完了，每跑一次，W,b都会更新，然后y跟着变，交叉煽也变，步长也每次下降（例子中没次下降0.1），期间可以调整各种参数，直到得出符合语预期的模型为止。
最后来一次test,就是拿新的数据来喂给它，看识别率了，识别率不满意时候，继续回去训练。
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("data/", one_hot=True)

sess = tf.InteractiveSession()

# Create the model
# 输入变量，把28*28的图片变成一维数组（丢失结构信息）
x = tf.placeholder(tf.float32, [None, 784])
# 权重矩阵，把28*28=784的一维输入，变成0-9这10个数字的输出
W = tf.Variable(tf.zeros([784, 10]))
# 偏置
b = tf.Variable(tf.zeros([10]))
# 核心运算，其实就是softmax（x*w+b）
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Define loss and optimizer
# 这个是训练集的正确结果
y_ = tf.placeholder(tf.float32, [None, 10])

# 交叉熵，作为损失函数
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 梯度下降算法，最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 初始化，在run之前必须进行的
# Train
tf.initialize_all_variables().run()
for i in range(10000):
    # 获取训练数据集的图片输入和正确表示数字
    batch_xs, batch_ys = mnist.train.next_batch(1000)
    # 运行刚才建立的梯度下降算法，x赋值为图片输入，y_赋值为正确的表示数字
    train_step.run({x: batch_xs, y_: batch_ys})
    # print(batch_xs,batch_ys)

# Test trained model
# tf.argmax获取最大值的索引。比较运算后的结果和本身结果是否相同。
# 这步的结果应该是[1,1,1,1,1,1,1,1,0,1...........1,1,0,1]这种形式。
# 1代表正确，0代表错误
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# tf.cast先将数据转换成float，防止求平均不准确。
# tf.reduce_mean由于只有一个参数，就是上面那个数组的平均值。
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 输出
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
