import numpy as np
import random
import math
import tensorflow as tf
import sys

def get_point_on_sphere(radius, dimensions):
    point = [random.uniform(0.0, radius) for i in range(dimensions)]    
    magnitude = 0.0
    for x in point:
        magnitude += x**2
    
    magnitude = math.sqrt(magnitude)

    return np.array([x / magnitude * radius for x in point])

def get_training_data(inner_radius, outer_radius, dimensions, count):
    x = []
    y = []
    for i in range(count):
        inner = random.choice([True, False])
        if inner:
            x.append(get_point_on_sphere(inner_radius, dimensions))
            y.append(np.array([1., 0.]))
        else:
            x.append(get_point_on_sphere(outer_radius, dimensions))
            y.append(np.array([0., 1.]))

    return (x, y)


def weight_variable(shape, trainable=True):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, trainable)

def bias_variable(shape, trainable=True):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, trainable)


dimensions = int(sys.argv[1])
x = tf.placeholder(tf.float32, [None, dimensions])
y_ = tf.placeholder(tf.float32, [None, 2])


W_fc1 = weight_variable([dimensions, 100])
b_fc1 = bias_variable([100])

h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

W_fc2 = weight_variable([100, 2])
b_fc2 = bias_variable([2])

y_model = tf.matmul(h_fc1, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_model))
train_step = tf.train.AdamOptimizer(5e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_model, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



adversarial_x = tf.Variable(np.reshape(get_point_on_sphere(1.0, dimensions), [1, dimensions]), dtype=tf.float32)

normalized_adversarial_x = tf.nn.l2_normalize(adversarial_x, 1)

W_fc1_fixed = weight_variable([dimensions, 100], False)
b_fc1_fixed = bias_variable([100], False)

h_fc1_fixed = tf.nn.relu(tf.matmul(normalized_adversarial_x, W_fc1_fixed) + b_fc1_fixed)

W_fc2_fixed = weight_variable([100, 2], False)
b_fc2_fixed = bias_variable([2], False)

y_model_fixed = tf.matmul(h_fc1_fixed, W_fc2_fixed) + b_fc2_fixed

adversarial_softmax = tf.nn.softmax(y_model_fixed)
adversarial_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_model_fixed))
adversarial_train_step = tf.train.AdamOptimizer(1e-4).minimize(adversarial_cross_entropy)
adversarial_correct_prediction = tf.equal(tf.argmax(y_model_fixed, 1), tf.argmax(y_, 1))
adversarial_accuracy = tf.reduce_mean(tf.cast(adversarial_correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5000):
        batch = get_training_data(1.0, 1.3, dimensions, 50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    test = get_training_data(1.0, 1.3, dimensions, 1000000)
    print('test accuracy %g' % accuracy.eval(feed_dict={x: test[0], y_: test[1]}))

    sess.run(tf.assign(W_fc1_fixed, W_fc1))
    sess.run(tf.assign(b_fc1_fixed, b_fc1))
    sess.run(tf.assign(W_fc2_fixed, W_fc2))
    sess.run(tf.assign(b_fc2_fixed, b_fc2))

    for i in range(5000):
        if i % 100 == 0:
            adversarial_input = normalized_adversarial_x.eval(feed_dict={x: [np.zeros([dimensions])], y_: [np.array([0, 1])]})
            softmax = adversarial_softmax.eval(feed_dict={x: [np.zeros([dimensions])], y_: [np.array([0, 1])]}) 
            train_accuracy = adversarial_accuracy.eval(feed_dict={x: [np.zeros([dimensions])], y_: [np.array([0, 1])]})
            print('step %d, softmax %s, training accuracy %g' % (i, softmax, train_accuracy))
        adversarial_train_step.run(feed_dict={x: [np.zeros([dimensions])], y_: [np.array([0, 1])]})