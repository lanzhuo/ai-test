#coding:utf-8
# 两层简单神经网络 （全连接）
import tensorflow as tf 

# 定义输入和参数
X  = tf.constant([[0.7, 0.5]])
W1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
W2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))

# 定义前向传播过程
a = tf.matmul(X, W1)
y = tf.matmul(a, W2)

# 用会话计算结果
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print "y in tf_3.py is: \n", sess.run(y)

