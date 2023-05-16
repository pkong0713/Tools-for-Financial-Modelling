#-------------------------------- Description --------------------------------
"""This file will analyze the stochastic matrix (aka Markov matrix),
given a dataframe of stock prices. The target input data is expected
to have the columns: <Date, Close, Volume>, where data should be sorted
in ascending order. This file will then convert the data into a column of data
with <Close> only for analysis. 

The parameters to this analysis will be:
    - the input file itself (decides number of trading days and security)
    - number of states: s
    
With these two variables, the model will come up with a Markov matrix with the
specified number of trading days and number of states (percentiles are equally
distributed). Target input == data.csv, Target output = Markov matrix A, and
the steady state vector pi"""
#---------------------------- Importing Libraries ----------------------------
import pandas as pd
import numpy as np

#------------------------ Loading and Processing data ------------------------
df = pd.read_csv(r'C:/Users/Philip/Desktop/quartz/data.csv')
df = df.reset_index()
df = df['Close']

#--------------------------- Setting the parameters --------------------------
s = 4

#--------------------------- Constructing the data ---------------------------
df_p = []

for i in range(len(df)-1):
    df_p.append((df[i+1]/df[i]-1)*100)

#-----------Parameter for dimension of probability transition matrix----------
percentiles = []
for i in range(s+1):
    percentiles.append(int(100*i/s))
    
border = []
for i in range(len(percentiles)):
    border.append(np.percentile(df_p,percentiles[i]))

#--------------------Creating the states and the transitions------------------
state_path = []

def is_between(a, x, b):
    return min(a, b) <= x <= max(a, b)
    
for i in df_p:
    for b in range(len(border)-1):
        if is_between(border[b],i,border[b+1]) == True:
            state_path.append(b)

state_path_1 = state_path[1:]
state_path = state_path[:len(df)-1]

transition = np.array([[state_path],[state_path_1]])
transition = transition.transpose()
transition = np.reshape(transition,(len(df)-1,2))

#-------------------------Creating Markov Matrix A----------------------------
markov_matrix = np.zeros((s,s))
for i in range(len(transition)):
    markov_matrix[state_path[i]][state_path_1[i]] = markov_matrix[state_path[i]][state_path_1[i]] + 1

markov_matrix_norm = np.zeros((s,s))
for i in range(s):
    for j in range(s):
        markov_matrix_norm[i][j] = markov_matrix[i][j]/sum(markov_matrix[i])

#----------------------Finding Steady State vector pi-------------------------
def steady_state_prop(p):
    dim = p.shape[0]
    q = (p - np.eye(dim))
    ones = np.ones(dim)
    q = np.c_[q, ones]
    QTQ = np.dot(q, q.T)
    bQT = np.ones(dim)
    return np.linalg.solve(QTQ, bQT)

steady_state_matrix = steady_state_prop(markov_matrix_norm)

#---------------------------------Result--------------------------------------
print(F"The number of states is {s}.")
print("The Steady State Probability is:")   
print(steady_state_matrix)
