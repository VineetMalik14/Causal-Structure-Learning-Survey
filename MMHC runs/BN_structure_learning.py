import random
import urllib.request
from os import path

import networkx as nx

from convert_csv_to_list import convert_csv_to_list
import pandas as pd
from graphviz import Digraph
from rpy2.robjects import pandas2ri
from lib.mmhc import mmhc

pandas2ri.activate()

import bnlearn
#base1, bnlearn1 = importr('base'), importr('bnlearn')
data = pd.read_csv('sim_data_10_100.csv')


dag_learned = mmhc(data, test = 'z-test')
nx.draw(dag_learned)
#adj_matrix = nx.to_numpy_array(dag_learned)
#print(adj_matrix)
#pd.DataFrame(dag_learned).to_csv('10_100_output.csv', index = False)