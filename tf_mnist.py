# coding: utf-8

# In[28]:


import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784
OUTPUT_NODE = 10

LAYER1_NODE = 500

BATCH_SIZE = 100

LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99

REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 150000
MOVING_AVGRAGE_DECAY = 0.99


# In[29]:


def inference(input_tensor, avg_class, weight1, biases1, weight2, biases2):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weight1) + biases1)
        return tf.matmul(layer1, weight2) + biases2  # 为啥不要激活函数
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weight1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weight2)) + avg_class.average(biases2)


# In[38]:


def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    weight1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    weight2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    y = inference(x, None, weight1, biases1, weight2, biases2)

    global_step = tf.Variable(0, trainable=False)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))

    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weight1) + regularizer(weight2)

    loss = cross_entropy_mean + regularization

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        init_op = tf.initialize_all_variables()
        sess.run(init_op)

        validate_feed = {x: mnist.validation.images, y_: convert_num2tensor(mnist.validation.labels)}
        print(validate_feed)
        test_feed = {x: mnist.test.images, y_: convert_num2tensor(mnist.test.labels)}
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("validate_acc=%f" % validate_acc)
                print("at step=%d" % i)
                xs, ys = mnist.train.next_batch(BATCH_SIZE)
                yt = convert_num2tensor(ys)
                sess.run(train_step, feed_dict={x: xs, y_: yt})

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("test_acc=%f" % validate_acc)


# In[45]:

def convert_num2tensor(arr):
    mat = np.zeros((len(arr), 10), dtype="float32")
    i = 0
    for num in arr:
        mat[i][num] = 1
        i += 1
    return mat


def main(argv):
    # mnist=input_data.read_data_sets(".//tfmnist",one_hot=True)
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")

    train(mnist)


if __name__ == "__main__":
    tf.app.run()
