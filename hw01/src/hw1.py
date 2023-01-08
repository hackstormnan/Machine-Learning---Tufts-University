'''
hw1.py
Author: TODO

Tufts CS 135 Intro ML

'''

import numpy as np

def split_into_train_and_test(x_all_LF, frac_test=0.5, random_state=None):
    len = x_all_LF.shape[0]
    N = int(np.ceil(frac_test * len))

    if (random_state == None) :
        random_state = np.random
    elif (isinstance(random_state, int)) :
        random_state = np.random.RandomState(random_state)

    data = random_state.permutation(x_all_LF)

    temp = len-N
    return data[:temp], data[temp:]



def calc_k_nearest_neighbors(data_NF, query_QF, K=1):

    q = query_QF.shape[0]
    f = query_QF.shape[1]
    ans = cdist(data_NF, query_QF).reshape(q, K, f)

    return ans