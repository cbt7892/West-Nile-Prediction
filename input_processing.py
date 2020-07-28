import numpy as np


def read_file(file_name, strings=False):
    # used for reading input files
    with open(file_name, "r") as file:
        data = []
        if not strings:
            for entry in file:
                data.append(float(entry.strip("\n")))
        else:
            for entry in file:
                data.append(entry.strip("\n"))
        return np.array(data)


def read_2d(file_name):
    # used for reading in a 2d array input file
    with open(file_name, "r") as file:
        data = np.zeros((7, 8))
        count = 0
        for line in file.readlines():
            vec = line.split("\t")
            for i in range(3, 11):
                data[count][i - 3] = float(vec[i])
            count += 1
        return data


def labels_to_array(labels):
    # convert GLOBE land cover labels to array for algorithm
    label_map = {'trees_canopycover': 0, 'bush/scrub': 1, 'grass': 2, 'cultivated vegetation': 3,
                 'water>treated pool': 4, 'water>lake/ponded/container': 5, 'water>rivers/stream': 6,
                 'water>irrigation ditch': 7, 'shadow': 8, 'unknown': 9, 'bare ground': 10, 'building': 11,
                 'impervious surface': 12}
    counts = np.zeros(13)
    for label in labels:
        counts[label_map[label.lower()]] += 1
    counts = np.divide(counts, np.sum(counts))
    return counts


def normalize(inputs):
    # normalize input features
    for column in range(inputs.shape[1]):
        mu = np.mean(inputs[:, column])
        sigma = np.std(inputs[:, column])
        if sigma != 0:
            inputs[:, column] = np.divide(inputs[:, column] - mu, sigma)
        else:
            continue
    return inputs


def assemble_data():
    temp_anomaly = read_file("data/annual_temp_anomaly.txt")
    dallas_temps = read_2d("data/dallas_temp.txt")
    globe_land_cover = labels_to_array(read_file("data/land_cover.txt", strings=True))
    satellite_data = np.load("data/satellite_data.npy").reshape(7, 10920)
    west_nile_counts = read_file("data/dallas_wnv.txt")

    all_inputs = np.zeros((7, 10942))
    for i in range(7):
        all_inputs[i][0] = temp_anomaly[i]
        all_inputs[i][1:9] = dallas_temps[i]
        all_inputs[i][9:22] = globe_land_cover[i]
        all_inputs[i][22:10942] = satellite_data[i]

    return [normalize(all_inputs), west_nile_counts]

