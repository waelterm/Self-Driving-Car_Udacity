import pickle
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#The purpose of this file is to analyze the traffic sign dataset


def decode_label(label_number):
    signnames = pd.read_csv('signnames.csv')
    label = signnames['SignName'][label_number]
    return label

if __name__ == '__main__':
    dataset_directory = 'dataset/'
    listOfFiles = os.listdir(dataset_directory)
    signnames = pd.read_csv('signnames.csv')
    print("Signnames Info: {}".format(signnames.info()))
    label_list = signnames['SignName']

    for file in listOfFiles:
        if file[-2:] != '.p':
            continue
        print("")
        print("Analyzing {}".format(file))
        print("===================================")
        data = pickle.load(open(dataset_directory + file, "rb"))

        print("Sizes")
        print("    Length : {}".format(len(data["sizes"])))
        print("    Example: {}".format(data["sizes"][146]))
        print("")

        print("Coords")
        print("    Length : {}".format(len(data["coords"])))
        print("    Example: {}".format(data["coords"][146]))
        print("")

        print("Features")
        print("    Length : {}".format(len(data["features"])))
        print("    Feature shape: {}".format(data["features"][146].shape))
        print("    Example has been saved as explore_dataset/" + file[:-2] + ".jpg")
        cv2.imwrite("explore_dataset/" + file[:-2] + ".jpg", cv2.cvtColor((data["features"][146]), cv2.COLOR_BGR2RGB));
        cv2.imshow("Sample Image", cv2.cvtColor(data["features"][146], cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        print("Labels")
        print("    Length : {}".format(len(data["labels"])))
        print("    Example: {}".format(data["labels"][146]))
        print("    Decoded Label: {}".format(decode_label(data["labels"][146])))
        label_cnt = np.zeros(43)
        for label in data["labels"]:
            label_cnt[label] += 1
        plt.bar(x = [str(i) for i in range(43)], height = label_cnt)
        plt.xlabel('Street Sign Label Number')
        plt.ylabel('Number of occurrences in dataset')
        plt.title('Label Distribution in "{}"'.format(file))


