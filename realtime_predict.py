from model import get_model, plot_results
from input_processing import assemble_data

features = assemble_data(chronological=True)
index = 10000  # signifies a date during the year, allows for real time predictions
x = features[0][:][0:index]
y = features[1]
x_train = features[0][0:5][0:index]  # train model on data up to that point for the year, feed in real time data
y_train = features[1][0:5]

model = get_model()
model.fit(x_train, y_train, epochs=50)
plot_results(model, x, y)

