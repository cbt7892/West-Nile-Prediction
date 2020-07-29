import csv
import numpy as np


def load_data():
    with open("data/GLOBEMeasurementData-14171.csv", "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        globe_data = np.zeros((42, 5))
        idx = 0
        count = 0
        for row in csv_reader:
            if idx == 0 or idx == 1:
                idx += 1
                continue
            lat = row[28]
            long = row[29]
            if -97.000482 <= float(long) <= -96.463632 and 32.613216 <= float(lat) <= 33.023937:
                if int(row[11][0:4]) > 2019 or int(row[11][0:4]) < 2013:  # beyond time range
                    continue
                if row[14] == "":
                    globe_data[count][0] = 0
                else:
                    globe_data[count][0] = float(row[14])  # mosquito count
                globe_data[count][1] = row[15] == "TRUE"
                globe_data[count][2] = row[17] == "TRUE"
                globe_data[count][3] = row[18] == "TRUE"  # useful data for this analysis
                globe_data[count][4] = float(row[30])

                count += 1
            idx += 1
        return globe_data
