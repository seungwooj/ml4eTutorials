
import tensorflow as tf
import matplotlib.pyplot as plt
#1. model 정의
#give data   
X = [1,2,3]
Y = [1,2,3]
#parameter선언
W = tf.placeholder(tf.float32)
#hypothesis for lineaer model : X * W
hypothesis = X * W
#cost(loss) function
cost = tf.reduce_mean(tf.square(hypothesis - Y))


#2.model 실행
#Launch the graph in a session and initialize global variables in the graph.
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#Variables for plotting cost function
W_val = []
cost_val = []
for i in range(-30, 50):
    feed_W = i * 0.1
    curr_cost, curr_W = sess.run([cost, W], feed_dict = {W: feed_W})
    W_val.append(curr_W)
    cost_val.append(curr_cost)

#show the cost function
plt.plot(W_val, cost_val)
plt.show()
