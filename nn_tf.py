import tensorflow as tf
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from matplotlib import pyplot as plt

def get_data_one_hot_encode(matrix_):
    one_hot = OneHotEncoder(sparse=False)
    label = LabelEncoder()

    classes = np.array([i[1] for i in matrix_])

    label.fit(classes)

    new_c = label.transform(classes)

    new_c = new_c.reshape(-1, 1)

    return one_hot.fit_transform(new_c)


path_ = '/train.txt'
matrix_train = np.genfromtxt(path_, delimiter=",", dtype=[('value', '4float64'), ('name', 'U16')])

X = np.array([i[0] for i in matrix_train])
Y = get_data_one_hot_encode(matrix_train)

input_shape = X.shape[1]

tr_x, tst_x, tr_y, tst_y = train_test_split(X, Y, random_state=415)


learning_rate = 0.175
n_class = 3
epochs = 150
n_hidden_1 = 6

cost_hist = np.empty(shape=[1], dtype=float)

x = tf.placeholder(tf.float32, [None, input_shape])
W = tf.Variable(tf.random_normal([input_shape, n_class]))
b = tf.Variable(tf.zeros(n_class))
y_ = tf.placeholder(tf.float32, [None, n_class])

mse_hist = []
acc_hist = []


def neural_network(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)

    output = tf.matmul(layer_1, weights['out']) + biases['out']

    return output


weights = {'h1': tf.Variable(tf.truncated_normal([input_shape, n_hidden_1])),
           'out': tf.Variable(tf.truncated_normal([n_hidden_1, n_class]))}
biases = {'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
          'out': tf.Variable(tf.truncated_normal([n_class]))}

init = tf.global_variables_initializer()

y = neural_network(x, weights, biases)

cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
tr_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_func)

sess = tf.Session()
sess.run(init)

for ep in range(epochs):
    sess.run(tr_step, feed_dict={x: tr_x, y_: tr_y})
    cost = sess.run(cost_func, feed_dict={x: tr_x, y_: tr_y})
    cost_hist = np.append(cost_hist, cost)
    correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    #
    pred_Y = sess.run(y, feed_dict={x: tst_x})
    mse = tf.reduce_mean(tf.square(pred_Y - tst_y))
    mse_ = sess.run(mse)
    mse_hist.append(mse_)
    accuracy = (sess.run(accuracy, feed_dict={x: tr_x, y_: tr_y}))
    acc_hist.append(accuracy)


print(mse_hist)
plt.plot(range(150),acc_hist,)
plt.plot(range(150),mse_hist,)

plt.show()
