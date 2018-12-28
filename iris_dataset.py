import tensorflow as tf
import numpy as np
import random

from sklearn.datasets import load_iris

def one_hot_encode(seq):
    one_hot = np.zeros((len(seq), max(seq) + 1), dtype=np.int32)
    for i, e in enumerate(seq):
        one_hot[i, e] += 1

    return one_hot

dataset = load_iris()
dataset = list(zip(dataset.data, dataset.target))

random.shuffle(dataset)

dataset = list(zip(*dataset))

train_x = dataset[0][:100]
train_y = one_hot_encode(dataset[1][:100])

test_x = dataset[0][100:]
test_y = one_hot_encode(dataset[1][100:])

# setup placeholders and network

features = tf.placeholder(tf.float32, [None, 4])
labels = tf.placeholder(tf.float32, [None, 3])

def network(x):
    y = tf.layers.dense(x, 10, tf.nn.relu)
    y = tf.layers.dense(y, 3)

    return y

predictions = tf.nn.softmax(network(features))

# loss and optimizers

loss = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(predictions), reduction_indices=[1]))
optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
correct = tf.equal(tf.argmax(predictions,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# train

num_step = 10000

session = tf.Session()

session.run(tf.global_variables_initializer())

for i in range(num_step):
    index = random.randint(0, len(train_x) - 21)
    _, l, a = session.run([optimizer, loss, accuracy], feed_dict={features: train_x[index: index+20], labels: train_y[index: index+20]})
    if i % 1000 == 0:
        print(f'loss: {l}; accuracy: {a}')

a = session.run(accuracy, feed_dict={features: test_x, labels: test_y})

print(f'test accuracy: {a}')