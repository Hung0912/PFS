import numpy as np
import random
import operator
import math
from copy import deepcopy
import time
import csv

def func(x):
    return 1 - x - np.square(1-np.sqrt(x))

print(func(1.12))