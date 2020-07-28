from input_processing import assemble_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l1
import matplotlib.pyplot as plt

features = assemble_data()

# unpack data
x = features[0]
y = features[1]

# model
model = Sequential()
model.add(Dense(256, input_dim=10942, kernel_initializer="normal", activation="relu", bias_regularizer=l1(0.01)))
model.add(Dense(64, kernel_initializer="normal", activation="relu", bias_regularizer=l1(0.01)))
model.add(Dense(8, kernel_initializer="normal", activation="relu", bias_regularizer=l1(0.01)))
model.add(Dense(1, kernel_initializer="normal"))
model.compile(loss="mean_squared_error", optimizer="adam")
model.fit(x, y, epochs=50)

# save model
model.save("wnv_prediction")

# graph results
years = range(2013, 2020)
plt.plot(years, y, "b-", label="CDC Data")
plt.plot(years, model.predict(x), "r-", label="Model Prediction")
plt.legend(loc="upper left")
plt.xlabel("Year")
plt.ylabel("Number of West Nile Virus Cases in Dallas, Texas")
plt.title("Model Prediction vs. Actual Data")
plt.show()







