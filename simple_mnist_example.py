import tensorflow as tf 

from tensorflow.examples.tutorials.mnist import input_data
print("Tensorflow version " + tf.__version__)

mnist = input_data.read_data_sets("data", one_hot=True, reshape=False)
print("shape of data:", mnist.test.images.shape)

def model():
    data = tf.placeholder(tf.float32, shape = [None, 28, 28, 1], name='input_data')
    conv1 = tf.layers.conv2d(data, kernel_size=5, use_bias=True, strides=(1,1), activation=tf.nn.relu, padding="SAME", filters=9 )
    # dp1   = tf.nn.dropout(conv1, keep_prob=0.75)

    
    conv2 = tf.layers.conv2d(conv1, kernel_size=5, use_bias=True, strides=(2,2), activation=tf.nn.relu, padding="SAME", filters=16 )
    # dp2   = tf.nn.dropout(conv2, keep_prob=0.75)

    conv3 = tf.layers.conv2d(conv2, kernel_size=5, use_bias=True, strides=(2,2), activation=tf.nn.relu, padding="SAME", filters=25 )
    # dp3   = tf.nn.dropout(conv3, keep_prob=0.75)

    flatten1 = tf.layers.flatten(conv3)
    fc1 = tf.layers.dense(flatten1, units=200, activation=tf.nn.relu, use_bias=True)
    dp1 = tf.nn.dropout(fc1, keep_prob=0.75)

    fc2 = tf.layers.dense(dp1, units=80, activation=tf.nn.relu, use_bias=True)
    dp2 = tf.nn.dropout(fc2, keep_prob=0.75)

    fc3 = tf.layers.dense(dp2, units=10, activation=tf.nn.relu, use_bias=True)

    result = tf.nn.softmax(fc3)
    return result



NUM_EPOCHS = 500
BATCH_SIZE = 100
lr = 0.0005

data_rows = mnist.test.images.shape[0]
print("data_rows:", data_rows)

num_steps = int(data_rows/BATCH_SIZE)
print("step size:", num_steps)

Y_ = tf.placeholder(tf.float32, shape=[None, 10])
Y = model()

loss = tf.nn.softmax_cross_entropy_with_logits(logits= Y, labels=Y_)
train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(NUM_EPOCHS):
        for step in range(num_steps):
            batch_X, batch_Y = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_step, feed_dict={'input_data:0': batch_X, Y_: batch_Y})
        
        pred = sess.run(accuracy, feed_dict={'input_data:0' : mnist.test.images, Y_ : mnist.test.labels})
        print(" Epoch: ", epoch, "accuracy: ", pred)

print("Training Complete .....................")



    

