import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle


def LeNet(x, keep_prob):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # TODO: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    F_W_c1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 6], mean=mu, stddev=sigma))
    F_b_c1 = tf.Variable(tf.zeros(6))
    strides = [1, 1, 1, 1]
    conv1 = tf.nn.conv2d(x, F_W_c1, strides, padding='VALID') + F_b_c1
    print(conv1.shape)

    # TODO: Activation.
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.dropout(conv1, keep_prob)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = 'VALID'
    conv1 = tf.nn.max_pool(conv1, ksize, strides, padding)

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    F_W_c2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], mean=mu, stddev=sigma))
    F_b_c2 = tf.Variable(tf.zeros(16))
    strides = [1, 1, 1, 1]
    conv2 = tf.nn.conv2d(conv1, F_W_c2, strides, padding='VALID') + F_b_c2

    # TODO: Activation.
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.dropout(conv2, keep_prob)
    print(conv2.shape)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = 'VALID'
    conv2 = tf.nn.max_pool(conv2, ksize, strides, padding)
    print(conv2.shape)

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    features = flatten(conv2)

    # TODO: Layer 3: Fully Connected. Input = 400. Output = 200.
    F_W_f1 = tf.Variable(tf.truncated_normal(shape=(400, 200), mean=mu, stddev=sigma))
    F_b_f1 = tf.Variable(tf.zeros(200))
    logits1 = tf.add(tf.matmul(features, F_W_f1), F_b_f1)

    # TODO: Activation.
    logits1 = tf.nn.relu(logits1)
    logits1 = tf.nn.dropout(logits1, keep_prob)


    # TODO: Layer 4: Fully Connected. Input = 200. Output = 84.
    F_W_f2 = tf.Variable(tf.truncated_normal(shape=[200, 84], mean=mu, stddev=sigma))
    F_b_f2 = tf.Variable(tf.zeros(84))
    logits2 = tf.add(tf.matmul(logits1, F_W_f2), F_b_f2)

    # TODO: Activation.
    logits2 = tf.nn.relu(logits2)
    logits2 = tf.nn.dropout(logits2, keep_prob)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    F_W_f3 = tf.Variable(tf.truncated_normal(shape=[84, 43], mean=mu, stddev=sigma))
    F_b_f3 = tf.Variable(tf.zeros(43))
    logits3 = tf.add(tf.matmul(logits2, F_W_f3), F_b_f3)
    return logits3




def evaluate(X_data, y_data, BATCH_SIZE,x, y, accuracy_operation):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

def run_session(num_labels, EPOCHS, BATCH_SIZE, X_train, y_train, X_validation, y_validation, keep_prob, rate = 0.001):
    x = tf.placeholder(tf.float32, (None, 32, 32, 3))
    y = tf.placeholder(tf.int32, (None))
    one_hot_y = tf.one_hot(y, num_labels)

    logits = LeNet(x, keep_prob)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=rate)
    training_operation = optimizer.minimize(loss_operation)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()
    validation_accuracies = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)

        print("Training...")
        print()
        for i in range(EPOCHS):
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

            validation_accuracy = evaluate(X_validation, y_validation, BATCH_SIZE ,x, y, accuracy_operation)
            print("EPOCH {} ...".format(i + 1))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
            validation_accuracies.append(validation_accuracy)
            print()
        saver.save(sess, './lenet')
        print("Model saved")
    return validation_accuracies

def restore_session(X_test, y_test):
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        test_accuracy = evaluate(X_test, y_test)
        print("Test Accuracy = {:.3f}".format(test_accuracy))