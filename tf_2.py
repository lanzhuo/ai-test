#coding:utf-8
# w = tf.Variable(tf.random_normal([2, 3], stddev=2, mean=0, seed=1))
#                      正态分布     2*3 矩阵  标准差    均值     随机种子
# tf.truncated_normal()  去掉过大偏离点的正态分布
# tf.random_uniform()    平均分布
#
# tf.zeros 全零数组      tf.zeros([3,2],int32) ->  [[0,0],[0,0],[0,0]]
# tf.ones  全1数组       tf.ones[[3,2],int32]  ->  [[1,1],[1,1],[1,1]]
# tf.fill  全定值数组     tf.fill([3,2],6)      ->  [[6,6],[6,6],[6,6]]
# tf.constant 直接给值    tf.constant([3,2,1])  ->  [3,2,1]
import tensorflow as tf

w = tf.Variable(tf.random_normal([2,3], stddev=2, mean=0, seed=1))
print w
