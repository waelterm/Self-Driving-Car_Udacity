# Import
import pickle
from neural_net import run_session
import csv


def import_dataset():
    """
    Imports test, training, and validation data
    return:  test, training, and validation dataset and classification
    """

    training_file = 'dataset/train.p'
    validation_file = 'dataset/valid.p'
    testing_file = 'dataset/test.p'

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
    """
    Evaluates the datasets and print information about them.
    :return:
    """
    ### Replace each question mark with the appropriate value.
    ### Use python, pandas or numpy methods rather than hard coding the results

    # TODO: Number of training examples
    assert (len(X_train) == len(y_train))
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
    Preprocesses the data by normalizing it
    :param X_train: Training set
    :param X_valid: Validation set
    :param X_test: Test set
    :return:
    """
    X_train_norm = (X_train.astype(float) - 128) / 128
    X_test_norm = (X_test.astype(float) - 128) / 128
    X_valid_norm = (X_valid.astype(float) - 128) / 128

    return X_train_norm, X_valid_norm, X_test_norm


if __name__ == '__main__':

    # Hyperparameter definition

    ### FIRST EXPLORATION TRAINING ###
    # LEARNING_RATES = [0.005, 0.0005, 0.001]
    # KEEP_PROBS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]
    # BATCH_SIZES = [128]

    ### SECOND EXPLORATION TRAINING
    # LEARNING_RATES = [0.001]
    # KEEP_PROBS = [0.9]
    # BATCH_SIZES = [32, 64, 128, 256, 512]

    ### THIRD EXPLORATION TRAINING
    # EPOCHS = 69
    # LEARNING_RATES = [0.0001]
    # KEEP_PROBS = [0.85, 0.9, 0.95]
    # BATCH_SIZES = [128]

    ### FINAL TRAINING
    EPOCHS = 69
    LEARNING_RATES = [0.001]
    KEEP_PROBS = [0.9]
    BATCH_SIZES = [128]

    # Loading dataset
    X_train, y_train, X_valid, y_valid, X_test, y_test = import_dataset()
    X_train_norm, X_valid_norm, X_test_norm = preprocess(X_train, X_valid, X_test)

    # Saving training stats in .csv file
    with open('accuracy_file.csv', mode='w', newline='') as accuracy_file:
        writer = csv.writer(accuracy_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = ["Epoch {}".format(i) for i in range(101)]
        header[0] = "Parameter Description"
        writer.writerow(header)
        # Iterate through combinations of hyperparameters.
        for LEARNING_RATE in LEARNING_RATES:
            for KEEP_PROB in KEEP_PROBS:
                for BATCH_SIZE in BATCH_SIZES:
                    print("Training with LR: {} KP: {} BS: {}".format(LEARNING_RATE, KEEP_PROB, BATCH_SIZE))
                    # Training model and saving accuracies after each epoch
                    validation_accuracies = run_session(num_labels=43, EPOCHS=EPOCHS, BATCH_SIZE=BATCH_SIZE,
                                                        X_train=X_train_norm, y_train=y_train,
                                                        X_validation=X_test_norm, y_validation=y_test,
                                                        rate=LEARNING_RATE, keep_prob=KEEP_PROB)
                    writer.writerow(["LR: {} KP: {} BS: {}".format(LEARNING_RATE, KEEP_PROB, BATCH_SIZE)] +
                                    [str(item) for item in validation_accuracies])
