import numpy as np
import pandas as pd
import seaborn as sns

from data_simulation import compute_dataset
from utils import dataset_overview, compute_indep_test, compute_features_usage_df

#                                E=0  E=1
probas_table_for_a1 = np.array([[0.9, 0.5],  # Xa1=0
                                [0.1, 0.5]]) # Xa1=1

#                                E=0  E=1
probas_table_for_a2 = np.array([[0.5, 0.7],  # Xa2=0
                                [0.5, 0.3]]) # Xa2=1

noise_on_y = 0.05
noise_on_Xc = 0.05

algos = ['ICSCM_stop_nomoreneg', 'ICSCM_stop_indepEY', 'SCM', 'DT']


df1 = compute_dataset(10000, noise_on_y=noise_on_y, noise_on_Xc=noise_on_Xc, probas_table_for_a1=probas_table_for_a1, probas_table_for_a2=probas_table_for_a2)
dataset_overview(df1)
print(compute_indep_test(df1, env_var=0, y_var=1))
feat_usage_df, perf_df = compute_features_usage_df(algos, df1, do_grid_search=False, param_grids=None)
sns.heatmap(feat_usage_df, annot=True, cmap='Greens')