import csv

header = ['class','sample']

rows = [[1, 2, 3]]

with open('test.csv', 'w') as f:
    f_csv = csv.writer(f, delimiter=' ')
    f_csv.writerow(header)
    f_csv.writerows(rows)

import json
import numpy as np
import os
import random
import scipy.io as sio
import tqdm

STEP = 256

def load_ecg_mat(ecg_file):
    return sio.loadmat(ecg_file)['val'].squeeze()

def load_all(data_path):
    label_file = os.path.join(data_path, "../REFERENCE-v3.csv")

    with open(label_file, 'r') as fid:
        records = [l.strip().split(",") for l in fid]
   

    dataset = []
    for record, label in tqdm.tqdm(records):
        ecg_file = os.path.join(data_path, record + ".mat")
        ecg_file = os.path.abspath(ecg_file)
        ecg = load_ecg_mat(ecg_file)
        num_labels = ecg.shape[0] / STEP
        dataset.append((ecg_file, [label]*num_labels))
    return dataset 

def split(dataset, dev_frac):
    dev_cut = int(dev_frac * len(dataset))
    random.shuffle(dataset)
    dev = dataset[:dev_cut]
    train = dataset[dev_cut:]
    return train, dev

def make_json(save_path, dataset):
    with open(save_path, 'w') as fid:
        for d in dataset:
            datum = {'ecg' : d[0],
                     'labels' : d[1]}
            json.dump(datum, fid)
            fid.write('\n')

def make_csv(save_path, dataset):
    import sys
    sys.path.append("../../ecg")
    import load
    with open(save_path, 'w') as fid:
        f_csv = csv.writer(fid, delimiter=' ')
        tag = ["label", "text"]
        f_csv.writerow(tag)
        for d in dataset:
            ecg_dir = d[0]
            data = load.load_ecg(ecg_dir)
            label = d[1][0]

            label_data = []
            label_data.append(label)
            for x in data:
                label_data.append(x)
            
            f_csv.writerow(label_data)
            
            





if __name__ == "__main__":
    random.seed(2018)

    dev_frac = 0.1
    data_path = "data/training2017/"
    dataset = load_all(data_path)

    train, dev = split(dataset, dev_frac)
    print(type(train))
    print(len(train))
    print(train[0])
    print(dev[0])
    make_csv("train.csv", train)

    # make_json("train.json", train)
    # make_json("dev.json", dev)

