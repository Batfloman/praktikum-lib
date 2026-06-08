from batfloman_praktikum_lib import graph, DataCluster
import numpy as np

data = DataCluster([
    { "a": 10, "b": 0},
    { "a": 11, "b": 1},
    { "a": 12, "b": 2},
    { "a": 13, "b": 3},
])
data2 = data.to_dataframe()

plot = graph.create_plot()
res = graph.plot(data, "a", "b")
res2 = graph.scatter(data, "a", "b")

graph.show(plot)