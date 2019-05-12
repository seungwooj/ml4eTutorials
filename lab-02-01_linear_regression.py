
# Lab 2 : linear regression
import tensorflow as tf

# 1) Build graph using TF operations (모델의 정 (모델의 실행 및 업데이트))

# # X and Y data
# x_train = [1, 2, 3]
# y_train = [1, 2, 3]

# model input and output : placeholders for X and Y
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])
# model parameter : shape, value 지정
W = tf.Variable(tf.random_normal([1]), name="weight") # random value, rank=1
b = tf.Variable(tf.random_normal([1]), name="bias")

# Our hypothesis XW+b
hypothesis = X * W + b
#cost/Loss function : mean of squares
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# minimize the cost function : GradientDiscent
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost) #cost function을 minimize하는 optimizer노드 : train

# 2), 3) Run/update graph and get results

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph. (W, b라는 variable을 run)
sess.run(tf.global_variables_initializer())

# # Fit the line : 앞서 graph에 지정한 GradientDiscent optimizer (train)을 실행
# for step in range(2001): #2000번 실행
#     sess.run(train)
#     if step % 20 == 0:
#         print(step, sess.run(cost), sess.run(W), sess.run(b)) #20번마다 print

# Fit the line
for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],
    feed_dict = {X: [1,2,3,4,5], Y: [2.1,3.1,4.1,5.1,6.1]})

    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)

# Testing our model
print(sess.run(hypothesis, feed_dict={X: [5]}))
print(sess.run(hypothesis, feed_dict={X: [2.5]}))
print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))
