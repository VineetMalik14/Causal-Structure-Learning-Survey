from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import IIDSimulation, DAG
from castle.algorithms import PC, Notears
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message="`np.int` is a deprecated alias for the builtin `int`")

def read_matrix(file_path):
    return  pd.read_csv(file_path).to_numpy()

def save_matrix(data, file_path):
    pd.DataFrame(data).to_csv(file_path, index=False)

FOLDER_PATH  = "./old/"

# data
X = read_matrix(os.path.join(FOLDER_PATH, "sim_data_70_7000_0.csv"))
true_causal_matrix = read_matrix(os.path.join(FOLDER_PATH, "causal_70_7000_0.csv"))

# structure learning
pc = Notears()
pc.learn(X)

# print(pc.causal_matrix)

# plot predict_dag and true_dag
GraphDAG(pc.causal_matrix, true_causal_matrix, 'result')

# calculate metrics
mt = MetricsDAG(pc.causal_matrix, true_causal_matrix)
print(mt.metrics)