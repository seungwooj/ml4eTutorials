
import tensorflow as tf

#1. model 정의
#give data
x_data = [1,2,3]
y_data = [1,2,3]
#변수 선언 : W만 있음.
W = tf.Variable(tf.random_normal([1]), name='weight')
#parameter 정의
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
#hypothesis for lineaer model : X * W
hypothesis = X * W
#cost(loss) function
cost = tf.reduce_mean(tf.square(hypothesis - Y))


# *** Gradient Descent 적용 방법
# # 1. Minimize: Gradient Descent using derivative: W -= Learning rate * derivative
# learning_rate = 0.1
# gradient = tf.reduce_mean((W * X - Y) * X)
# descent = W - learning_rate * gradient
# update = W.assign(descent) # W를 업데이트

# 2. Minimize: Gradient descent magic
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

#2.model 실행
#Launch the graph in a session and initialize global variables in the graph.
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(201):
    # sess.run(update, feed_dict={X: x_data, Y: y_data})
    # print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))
    cost_val, W_val, _ = sess.run([cost, W, train], feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        print(step, cost_val, W_val)
