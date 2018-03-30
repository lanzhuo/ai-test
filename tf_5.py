#coding:utf-8
# 两层简单神经网络 （全连接）
import tensorflow as tf 

# 定义输入和参数
# 用placeholder实现输入定义  （sess.run中喂多组数据）
X  = tf.placeholder(tf.float32, shape=(None,2))
W1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
W2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))

# 定义前向传播过程
a = tf.matmul(X, W1)
y = tf.matmul(a, W2)

# 用会话计算结果
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print "y in tf_5.py is: \n", sess.run(y, feed_dict={X: [[0.7,0.5], [0.2,0.3], [0.3,0.4], [0.4,0.5]]})
    print "W1:\n", sess.run(W1)
    print "W2:\n", sess.run(W2)

