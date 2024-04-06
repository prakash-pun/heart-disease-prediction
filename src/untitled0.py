from visualize import Visualizer
from data_loading import DataInitializer
#Modules
reader = DataInitializer()
plotter = Visualizer()

X_train, X_test, y_train, y_test = reader.split_data()
plotter.barplot(X_train)
plotter.barplot(y_train)
plotter.scatter(X_train,y_train)
plotter.heatmap(X_train, y_train)


