import tensorflow as tf
from neural_net import LeNet
from neural_net import evaluate
from classifier import import_dataset
from classifier import preprocess

# Load and normalize dataset
X_train, y_train, X_valid, y_valid, X_test, y_test = import_dataset()
X_train_norm, X_valid_norm, X_test_norm = preprocess(X_train, X_valid, X_test)

#Prepare to load trained network
save_file = './saved_models/lenet'
keep_prob = 1
tf.reset_default_graph()
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

logits = LeNet(x, keep_prob)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

BATCH_SIZE = 128

with tf.Session() as sess:
    saver.restore(sess, save_file)
    test_accuracy = evaluate(X_test_norm, y_test, BATCH_SIZE , x , y, accuracy_operation)
    train_accuracy = evaluate(X_train_norm, y_train, BATCH_SIZE, x, y, accuracy_operation)
    validation_accuracy = evaluate(X_valid_norm, y_valid, BATCH_SIZE, x, y, accuracy_operation)
    print("Test Accuracy: {:.2%}%".format(test_accuracy))
    print("Train Accuracy: {:.2%}%".format(train_accuracy))
    print("Validation Accuracy: {:.2%}%".format(validation_accuracy))

