#coding:utf-8

# -- 预测酸奶销量，使用自定义损失函数 -- 
# -- 更改利润和成本 --

# 两层简单神经网络 （全连接）
import tensorflow as tf 
import numpy as np 

BATCH_SIZE = 8
SEED = 23455

COST   = 9
PROFIT = 1

# 基于seed产生随机数
rdm = np.random.RandomState(SEED)
# 随机数返回32行２列的矩阵　表示32组　体积和重量　作为输入数据集
X  = rdm.rand(32,2)
Y_ = [[x1+x2+(rdm.rand()/10.0-0.05)] for (x1, x2) in X]
print "X:\n", X 
print "Y_:\n", Y_

# 1 定义神经网络的输入、参数和输出，定义前向传播过程
x  = tf.placeholder(tf.float32, shape=(None,2))
y_ = tf.placeholder(tf.float32, shape=(None,1))

w1 = tf.Variable(tf.random_normal([2,1], stddev=1, seed=1))

y = tf.matmul(x, w1)

# 2 定义损失函数及反向传播方法
#loss = tf.reduce_mean(tf.square(y-y_))
loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y-y_)*COST, (y_-y)*PROFIT))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
# train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
# train_step = tf.train.MomentumOptimizer(0.001, 0.9).minimize(loss)

# 3 生成会话，训练STEPS轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 训练模型
    STEPS = 20000
    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
        if i % 500 == 0:
            print "After %d training step(s), w1 is: " % (i)
            # 输出训练后的参数取值
            print sess.run(w1), "\n"
    
    print "Final w1 is: \n", sess.run(w1)

