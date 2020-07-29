from input_processing import assemble_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l1
import matplotlib.pyplot as plt


def get_model():
    nn = Sequential()
    nn.add(Dense(256, input_dim=11152, kernel_initializer="normal", activation="relu", bias_regularizer=l1(100)))
    nn.add(Dense(64, kernel_initializer="normal", activation="relu", bias_regularizer=l1(100)))
    nn.add(Dense(8, kernel_initializer="normal", activation="relu", bias_regularizer=l1(100)))
    nn.add(Dense(1, kernel_initializer="normal"))
    nn.compile(loss="mean_squared_error", optimizer="adam")
    return nn


def plot_results(model, x, y):
    years = range(2013, 2020)
    plt.plot(years, y, "b-", label="CDC Data")
    plt.plot(years, model.predict(x), "r-", label="Model Prediction")
    plt.legend(loc="upper left")
    plt.xlabel("Year")
    plt.ylabel("Number of West Nile Virus Cases in Dallas, Texas")
    plt.title("Model Prediction vs. Actual Data")
    plt.show()


def main():
    features = assemble_data()

    # unpack data
    x = features[0]
    x_train = x[0:5]
    x_test = x[5:]
    y = features[1]
    y_train = y[0:5]
    y_test = y[5:]

    # model
    model = get_model()
    model.fit(x_train, y_train, epochs=50)

    # save model
    model.save("wnv_prediction")

    # graph results
    plot_results(model, x, y)


if __name__ == "__main__":
    main()

