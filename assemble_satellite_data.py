import h5py
import os
import re
import numpy as np


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


directory = "/users/coletramel/pycharmprojects/mosquitoes/data/AMSR DATA/"

with open("attrs.txt", "r") as f:
    attrs = f.readlines()

files = []

for file in os.listdir(directory):
    if file.endswith(".he5"):
        files.append(file)
files.sort(key=natural_keys)  # sort files by time

satellite_data = np.zeros((7, 78, 35, 4))

# assemble satellite data into numpy array
index = 0
for name in files:
    with h5py.File(directory + name, "r") as f:
        year = int(name[20:24])

        dset = f["HDFEOS"]["POINTS"]["AMSR-2 Level 2 Land Data"]["Data"]["Combined NPD and SCA Output Fields"]
        data = dset[:]

        lat = data["Latitude"]
        long = data["Longitude"]
        long_indices = np.where(np.logical_and(long >= -97.000482, long <= -96.463632))  # dallas longitude
        lat_indices = np.where(np.logical_and(lat <= 33.023937, lat >= 32.613216))  # dallas latitude
        dallas = np.intersect1d(long_indices, lat_indices)
        if len(dallas) != 4:
            continue  # when satellite doesn't have 4 observations, skip

        features = []
        for attr in attrs:
            features.append(data[attr.strip("\n")][dallas])
        features = np.array(features)
        satellite_data[-1 * (year - 2019)][index] = features
        index += 1
        index %= 78

np.save("satellite_data.npy", satellite_data)
