import tensorflow as tf
import cv2
from neural_net import LeNet
import numpy as np
import os
import pandas as pd

# Import image names
images = []
for subdir, dirs, files in os.walk('./my_test_images/Formatted'):
    for file in files:
        if file[-3:] == 'png':
            images.append(file)

# Load images, resize and add to image list X_train
X_train = []
raw_images = []
for image in images:
    img = cv2.imread('my_test_images\\Formatted\\' + image)
    raw_images.append(img)
    img = cv2.resize(img, (32, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_array = np.empty((1,32,32,3))
    image_array[0] = np.array(img)
    print(image_array)
    #cv2.imshow("Image1", img)
    #cv2.waitKey(0)
    image_array = (image_array.astype(float) - 128) / 128
    X_train.append(image_array)

# Prepare to load saved model
save_file = './saved_models/lenet'
tf.reset_default_graph()
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
logits = LeNet(x, 1)
saver = tf.train.Saver()


with tf.Session() as sess:
    # Restore trained network
    saver.restore(sess, save_file)

    # Import Sign decoding file
    signnames = pd.read_csv('signnames.csv')  # doctest: +SKIP
    for i in range(len(X_train)):
        print("Image File: {}".format(images[i]))
        # Classify and annotate each image
        #print(X_train[i].shape)
        #print(X_train[i])
        my_logits = sess.run(logits, feed_dict={x: X_train[i]})
        predictions = tf.nn.softmax(my_logits)
        index = sess.run(tf.argmax(predictions, axis=1))
        #print("Index: {}".format(index))
        my_predictions = sess.run(predictions).tolist()[0]
        #print(my_predictions)
        idx = my_predictions.index(max(my_predictions))
        value = my_predictions[idx]
        my_predictions[idx] = 0
        sign_type = signnames.SignName[idx]
        # Print next guesses
        print("Guess #{}: {} , Confidence: {}".format(1, sign_type, value))
        for j in range(4):
            other_idx = my_predictions.index(max(my_predictions))
            other_value = my_predictions[other_idx]
            my_predictions[other_idx] = 0
            print("Guess #{}: {} , Confidence: {}".format(j+2, signnames.SignName[other_idx], other_value))

        # Annotate images
        #print("Value: {}% Index: {} Sign Type: {}".format(value*100, idx, sign_type))
        cv2.imshow("Image1", raw_images[i])
        cv2.waitKey(1000)
        cv2.putText(raw_images[i], "Sign_Type: {}".format(sign_type), org=(50, 75), \
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2,
                    lineType=cv2.LINE_AA)
        cv2.putText(raw_images[i], "Confidence: {0:.2f}%".format(value * 100), org=(50, 125), \
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2,
                    lineType=cv2.LINE_AA)
        cv2.imshow("Image1", raw_images[i])
        cv2.waitKey(0)
print('Finished')