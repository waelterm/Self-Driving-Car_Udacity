import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
from neural_net import run_session
import csv

def import_dataset():
    # Load pickled data
    import pickle

    # TODO: Fill this in based on where you saved the training and testing data

    training_file = 'dataset/train.p'
    validation_file ='dataset/valid.p'
    testing_file = 'dataset/valid.p'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test, y_test = test['features'], test['labels']

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def analyze_data(X_train, y_train, X_valid, y_valid, X_test, y_test):
    ### Replace each question mark with the appropriate value.
    ### Use python, pandas or numpy methods rather than hard coding the results

    # TODO: Number of training examples
    assert(len(X_train) == len(y_train))
    n_train = len(X_train)

    # TODO: Number of validation examples
    assert (len(X_valid) == len(y_valid))
    n_validation = len(X_valid)

    # TODO: Number of testing examples.
    assert (len(X_test) == len(y_test))
    n_test = len(X_test)

    # TODO: What's the shape of an traffic sign image?
    image_shape = X_train[0].shape

    # TODO: How many unique classes/labels there are in the dataset.
    n_classes = len(dict.fromkeys(y_train))

    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)

def preprocess(X_train, X_valid, X_test):
    """
    WORK
    :param X_train:
    :param X_valid:
    :param X_test:
    :return:
    """
    X_train_norm =  (X_train.astype(float) - 128) / 128
    X_test_norm = (X_test.astype(float) - 128) / 128
    X_valid_norm = (X_valid.astype(float) - 128) / 128

    return X_train_norm, X_valid_norm, X_test_norm


if __name__ == '__main__':
    X_train, y_train, X_valid, y_valid, X_test, y_test = import_dataset()
    analyze_data(X_train, y_train, X_valid, y_valid, X_test, y_test)
    X_train_norm, X_valid_norm, X_test_norm = preprocess(X_train, X_valid, X_test)
    #print(X_train_norm)
    EPOCHS = 100
    BATCHS_SIZE = 128
    LEARNING_RATES = [0.005, 0.0005]
    KEEP_PROBS = [0.5, 0.6, 0.7, 0.8, 0.9]
    import csv

    with open('accuracy_file.csv', mode='w', newline='') as accuracy_file:
        writer = csv.writer(accuracy_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = ["Epoch {}".format(i) for i in range(101)]
        header[0] = "Parameter Description"
        writer.writerow(header)
        for LEARNING_RATE in LEARNING_RATES:
            for KEEP_PROB in KEEP_PROBS:
                print("Training with LR: {} KP: {}".format(LEARNING_RATE, KEEP_PROB))
                validation_accuracies = run_session(num_labels = 43, EPOCHS = EPOCHS, BATCH_SIZE = BATCHS_SIZE,
                                                    X_train = X_train_norm, y_train = y_train,
                                                    X_validation = X_valid_norm, y_validation = y_valid,
                                                    rate=LEARNING_RATE, keep_prob = KEEP_PROB )
                writer.writerow(["LR: {} KP: {}".format(LEARNING_RATE, KEEP_PROB)] +
                                [str(item) for item in validation_accuracies])


    # To do: Add additional layer - NOT NECESSARY
    # Add dropout - DONE
    # Add zoom, rotation, etc. to dataset
    # write gridsearch configurator for training - DONE
    # Run on GPU
