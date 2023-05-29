#-------------------------------- Description --------------------------------
"""This file will analyze the stochastic matrix (aka Markov matrix),
given a dataframe of stock prices. The target input data is expected
to have the column <Close>

The parameters to this analysis will be:
    - the input file itself (decides number of trading days and security)
    - the 'borders' of the states, in percentages
    
With these two variables, the model will come up with a Markov matrix with the
specified number of trading days and number of states (percentiles are equally
distributed). Target input = data.csv, Target output = Markov matrix A, and
the steady state vector pi"""
#---------------------------- Importing Libraries ----------------------------
import pandas as pd
import numpy as np

#------------------------ Loading and Processing data ------------------------
df = pd.read_csv(r'C:/Users/Philip/Desktop/quartz/data.csv')
df = df.reset_index()
df = df['Close']

#--------------------------- Constructing the data ---------------------------
df_p = []

for i in range(len(df)-1):
    df_p.append((df[i+1]/df[i]-1)*100)

#-----------Parameter for dimension of probability transition matrix----------
border = [min(df_p), 5, 0, -4, max(df_p)]
border.sort()
s = len(border)-1

#--------------------Creating the states and the transitions------------------
state_path = []

def is_between(a, x, b):
    return min(a, b) <= x <= max(a, b)
    
for i in df_p:
    for b in range(s):
        if is_between(border[b],i,border[b+1]) == True:
            state_path.append(b)

state_path_1 = state_path[1:]
state_path = state_path[:len(state_path)-1]

transition = np.array([[state_path],[state_path_1]])
transition = transition.transpose()
transition = np.reshape(transition,(len(state_path),2))

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
