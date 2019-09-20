import numpy as np
import random
import operator
import math
import time

def random_with_sum(suma):
    a = np.random.rand(k,1)
    summation = np.sum(a, axis= 0)
    a = (a / summation) * suma
    return a

def init_membership_values():
    #Each data point
    while 1 :
        #random e 
        e_arr = np.random.rand(k,1)
        sum_e_arr = np.sum(e_arr, axis= 0)

        #calcu n
        sum_n_arr = 1 - sum_e_arr / k
        n_arr = random_with_sum(sum_n_arr)

        #calcu u
        rando = random_with_sum(1)
        u_arr = rando / (2 - e_arr)
    
        s = u_arr + e_arr + n_arr

        if (np.all(s<=1, axis= 0) & np.all(e_arr>=0, axis= 0) & np.all(n_arr>=0, axis= 0) & np.all(u_arr>=0, axis= 0)):
            break  
    result_matrix = np.concatenate((u_arr, n_arr, e_arr), axis= 1)
    return result_matrix

    
def init_membership_matrix():
    membership_matrix = np.zeros((n,5,3))
    for i in range(n):
        membership_matrix[i] = init_membership_values()
    # # print(membership_matrix)
    print("init done")
    return membership_matrix

membership_matrix = init_membership_matrix