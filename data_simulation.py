# *********************** DEFINE DATA ***********************

import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



random.seed(7)

def compute_y(a1, a2, noise_on_y):
    y_theory = int((a1 == 1) and (a2 == 1))
    #y_theory = int((a1 == 1) or (a2 == 1)) #disjunction
    r_value = random.random()
    if (r_value < noise_on_y):
        random_y = random.random()
        y = int(random_y < 0.5)
    else:
        y = y_theory
    return y

def compute_Xc_v1(y, e, noise_on_Xc):
    Xc_theory = int(y)
    r_value = random.random()
    if (r_value < noise_on_Xc):
        if e == 0:
            Xc = 1
        else:
            Xc = 0
    else:
        Xc = Xc_theory
    return Xc

def compute_Xc_v2(y, e, noise_on_Xc):
    Xc_theory = int(y)
    r_value = random.random()
    if (r_value < noise_on_Xc[e]):
        random_Xc = random.random()
        Xc = int(random_Xc < 0.5)
    else:
        Xc = Xc_theory
    return Xc

# faire 
# niveau de bruit sur Xc
# niveau de bruit entre Xa et Y

def compute_dataset(n_samples_per_env, noise_on_y, noise_on_Xc, probas_table_for_a1, probas_table_for_a2):
    data = []
    for e in [0,1]:
        for i in range(n_samples_per_env):
            b0_rand = random.random()
            a1_rand = random.random()
            a2_rand = random.random()
            b0 = 0 if b0_rand < 0.5 else 1
            a1 = 0 if a1_rand < probas_table_for_a1[0,e] else 1
            a2 = 0 if a2_rand < probas_table_for_a2[0,e] else 1
            y = compute_y(a1, a2, noise_on_y)
            Xc = compute_Xc_v1(y, e, noise_on_Xc)
            data.append([e, y, b0, a1, a2, Xc])
    df1 = pd.DataFrame(data, columns=['E', 'Y', 'Xb0', 'Xa1', 'Xa2', 'Xc'])
    return df1
