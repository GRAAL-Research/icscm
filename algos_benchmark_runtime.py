# *********************** DEFINE DATA ***********************

import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#                                E=0  E=1
probas_table_for_a1 = np.array([[0.9, 0.5],  # Xa1=0
                                [0.1, 0.5]]) # Xa1=1

#                                E=0  E=1
probas_table_for_a2 = np.array([[0.5, 0.7],  # Xa2=0
                                [0.5, 0.3]]) # Xa2=1

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

def compute_dataset(n_samples_per_env, n_random_variables, noise_on_y, noise_on_Xc, random_seed=11):
    random.seed(random_seed)
    data = []
    for e in [0,1]:
        for i in range(n_samples_per_env):
            a1_rand = random.random()
            a2_rand = random.random()
            a1 = 0 if a1_rand < probas_table_for_a1[0,e] else 1
            a2 = 0 if a2_rand < probas_table_for_a2[0,e] else 1
            y = compute_y(a1, a2, noise_on_y)
            Xc = compute_Xc_v1(y, e, noise_on_Xc)
            random_variables_list, random_variables_list_names = [], []
            for j in range(n_random_variables):
                b0_rand = random.random()
                b0 = 0 if b0_rand < 0.5 else 1
                random_variables_list.append(b0)
                random_variables_list_names.append('Xb' + str(j))
            data.append([e, y] + random_variables_list + [a1, a2, Xc])
    df1 = pd.DataFrame(data, columns=['E', 'Y'] + random_variables_list_names + ['Xa1', 'Xa2', 'Xc'])
    return df1



param_grids = {
    'robustSCM': {'p': [0.5, 0.75, 1.0, 2.5, 5],},
    'residualSCM': {'p': [0.5, 0.75, 1.0, 2.5, 5], 'model_type': ['conjunction'], 'threshold': [0.001, 0.05, 0.1], 'resample_rules' : [False]},
    'residualSCM resample': {'p': [0.5, 0.75, 1.0, 2.5, 5], 'model_type': ['conjunction'], 'threshold': [0.001, 0.05, 0.1], 'resample_rules' : [True]},
    'ICSCM': {'p': [0.1, 0.5, 0.75, 1.0, 2.5, 5, 10], 'model_type': ['conjunction'], 'stopping_method': ['independance_y_e']},
    'ICSCM_nomoreneg': {'p': [0.1, 0.5, 0.75, 1.0, 2.5, 5, 10], 'model_type': ['conjunction'], 'stopping_method': ['no_more_negatives']},
    'ICSCM_indepEY': {'p': [0.1, 0.5, 0.75, 1.0, 2.5, 5, 10], 'model_type': ['conjunction'], 'stopping_method': ['independance_y_e']},
    'SCM': {'p': [0.1, 0.5, 0.75, 1.0, 2.5, 5, 10], 'model_type': ['conjunction']}, 
    'ICP+SCM': {'p': [0.1, 0.5, 0.75, 1.0, 2.5, 5, 10], 'model_type': ['conjunction']}, 
    'SCM 1rule': {'p': [0.5, 0.75, 1.0, 2.5, 5], 'model_type': ['conjunction'], 'max_rules': [1]}, 
    'DT':
        {
            #'criterion' : ['gini', 'entropy'],
            #'splitter' : ['best', 'random'],
            'min_samples_split' : [2, 0.01, 0.05, 0.1, 0.3],
            #'min_samples_leaf' : [1, 2, 0.01, 0.05, 0.1, 0.3, 0.5],
            'max_depth' : [1, 2, 3, 4, 5, 10],
            #'max_features' : [0.01, 0.1, 0.25, 0.5, 0.75, 1.0],
        },
    'ICP+DT': {'min_samples_split' : [2, 0.01, 0.05, 0.1, 0.3], 'max_depth' : [1, 2, 3, 4, 5, 10]}, 
    'RF':
        {
            'n_estimators' : [1, 10, 100, 250],
            #'criterion' : ['gini', 'entropy'],
            #'bootstrap' : [True, False],
            'min_samples_split' : [2, 0.01, 0.05, 0.1, 0.3],
            #'min_samples_leaf' : [1, 2, 0.01, 0.05, 0.1, 0.3, 0.5],
            'max_depth' : [1, 2, 3, 5, 10],
            #'max_features' : ['log2', 'sqrt', 0.1, 0.25, 0.5, 0.75, None],
        }, 
    'regression': {'penalty': ['l1', 'l2', 'elasticnet', None], 'C': np.logspace(-5, 5, 11)}, 
    'SVM linear': {'C': np.logspace(-5, 5, 11)}, 
    #'SVM linear': {'C': np.logspace(-5, 5, 11)}, 
}

################## reload home algos if updated
import importlib

import icscm
import icpscm
import icpdt
importlib.reload(icscm)
importlib.reload(icpscm)
importlib.reload(icpdt)
from icscm import InvariantCausalSCM
from icpscm import InvariantCausalPredictionSetCoveringMachine
from icpdt import InvariantCausalPredictionDecisionTree

def init_model(algo):
    if algo == 'SCM':
        model = SetCoveringMachineClassifier(random_state=11)
    elif algo == 'SCM 1rule':
        model = SetCoveringMachineClassifier(max_rules=1, random_state=11)
    elif algo == 'RF':
        model = RandomForestClassifier(random_state=11)
    elif algo == 'DT':
        model = DecisionTreeClassifier(random_state=11)
    elif algo == 'SVM linear':
        model = SVC(kernel='linear', random_state=11)
    elif algo == 'SVM poly':
        model = SVC(kernel='poly', random_state=11)
    elif algo == 'SVM rbf':
        model = SVC(kernel='rbf', random_state=11)
    elif algo == 'regression':
        model = LogisticRegression(random_state=11)
    elif algo == 'ICSCM':
        model = InvariantCausalSCM(threshold=0.05, random_state=11)
    elif algo == 'ICSCM_nomoreneg':
        model = InvariantCausalSCM(threshold=0.05, stopping_method='no_more_negatives', random_state=11)
    elif algo == 'ICSCM_indepEY':
        model = InvariantCausalSCM(threshold=0.05, stopping_method='independance_y_e', random_state=11)
    elif algo == 'FSCM1':
        model = FlorenceSetCoveringMachineClassifier1(random_state=11)
    elif algo == 'FSCM1min':
        model = FlorenceSetCoveringMachineClassifier1min(random_state=11)
    elif algo == 'FSCM2':
        model = FlorenceSetCoveringMachineClassifier2(random_state=11)
    elif algo == 'ICP+SCM':
        model = InvariantCausalPredictionSetCoveringMachine(threshold=0.05, random_state=11)
    elif algo == 'ICP+DT':
        model = InvariantCausalPredictionDecisionTree(threshold=0.05, random_state=11)
    else:
        raise ValueError('unknown algo', algo)
    return model


# compute_features_usage_df
from datetime import datetime

from pyscm import SetCoveringMachineClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def compute_features_usage_df(algos, generated_df_list, do_grid_search=False, param_grids=None):
    features_usage_df_list = []
    perf_df = pd.DataFrame(columns=['algo', 'score', 'metric', 'type', 'split'])
    row_i = 0
    for repetition in range(len(generated_df_list)):
        print('  repetition', repetition)
        df2 = generated_df_list[repetition].copy()
        y = df2['Y'].values
        del df2['Y']
        X = df2.values
        features_names = list(df2)
        true_causal_features = ['Xa1', 'Xa2']
        true_causal_features_vector = [int(f in true_causal_features) for f in features_names]
        #print('features_names', features_names)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11+repetition)
        features_usage_df = pd.DataFrame(columns=algos, index=features_names)
        for algo in algos:
            print(algo)
            model = init_model(algo)
            # grid search for best parameters
            if do_grid_search and (algo in param_grids):
                hyperparameters = param_grids[algo]
                grid = GridSearchCV(
                    estimator=model,
                    param_grid=hyperparameters,
                    verbose=2,
                    n_jobs=-1,
                )
                grid_result = grid.fit(X_train, y_train)
                tuned_hyperparameters = grid_result.best_params_
                print('tuned_hyperparameters', tuned_hyperparameters)
                #cv_results_df = pd.DataFrame(grid_result.cv_results_)  # convert GS results to a pandas dataframe
                model.set_params(**tuned_hyperparameters)  # set best params
                model.set_params(random_state=11)  # set random state
            # time 1 :
            t1 = datetime.now()
            model.fit(X_train, y_train)
            # time after fited :
            t2 = datetime.now()
            test_pred = model.predict(X_test)
            train_pred = model.predict(X_train)
            perf_df.loc[row_i] = [algo, accuracy_score(y_test, test_pred), 'accuracy', 'test', repetition]
            row_i += 1
            perf_df.loc[row_i] = [algo, accuracy_score(y_train, train_pred), 'accuracy', 'train', repetition]
            row_i += 1
            if 'SCM' in algo and algo != 'ICP+SCM':
                features_used = [0]*len(features_names)
                if hasattr(model, 'rule_importances'):
                    for i in range(len(model.rule_importances)):
                        feat_name = features_names[model.model_.rules[i].feature_idx]
                        if model.rule_importances[i] > 0:
                            features_used[model.model_.rules[i].feature_idx] = model.rule_importances[i]              
            elif algo in ['RF', 'DT', 'ICP+DT', 'ICP+SCM']:
                features_used = model.feature_importances_
            elif algo == 'regression':
                positive_coeffs = [abs(coeff) for coeff in model.coef_[0]/sum(model.coef_[0])]
                features_used = positive_coeffs/sum(positive_coeffs)
            elif 'SVM' in algo:
                features_used = np.abs(model.coef_.flatten())
                features_used = features_used/sum(features_used)
            else:
                raise Exception('algo not implemented')
            features_used_binary = [1 if f > 0 else 0 for f in features_used]
            features_usage_df[algo] = features_used_binary
            causal_score = int(features_used_binary == true_causal_features_vector)
            print('features_used_binary', features_used_binary)
            print('true_causal_features_vector', true_causal_features_vector)
            perf_df.loc[row_i] = [algo, causal_score, '01 loss', 'causal', repetition]
            row_i += 1
            perf_df.loc[row_i] = [algo, (t2 - t1).total_seconds(), 't2-t1', 'fit time', repetition]
            row_i += 1
        features_usage_df['feature'] = features_usage_df.index
        features_usage_df['split'] = [repetition]*features_usage_df.shape[0]
        features_usage_df_list.append(features_usage_df)
    features_usage_df = pd.concat(features_usage_df_list, axis=0)
    return features_usage_df, perf_df

def dataset_overview(df):
    n_env = len(df['E'].unique())
    print('Dataset overview:')
    print(len(df), 'samples')
    print('   # of positive samples: {} ({} %)'.format(len(df[df['Y'] == 1]), 100*len(df[df['Y'] == 1])/len(df)))
    print('   # of negative samples: {} ({} %)'.format(len(df[df['Y'] == 0]), 100*len(df[df['Y'] == 0])/len(df)))
    for e in df['E'].unique():
        print('   e = {}  # of samples : {} ({} %)'.format(e, len(df[df['E'] == e]), 100*len(df[df['E'] == e])/len(df)))
        print('     # of positive samples in E{}: {} ({} %)'.format(e, len(df[(df['E'] == e) & (df['Y'] == 1)]), 100*len(df[(df['E'] == e) & (df['Y'] == 1)])/len(df[df['E'] == e])))
        print('     # of negative samples in E{}: {} ({} %)'.format(e, len(df[(df['E'] == e) & (df['Y'] == 0)]), 100*len(df[(df['E'] == e) & (df['Y'] == 0)])/len(df[df['E'] == e])))


noise_on_y = 0.05
noise_on_Xc = 0.05

random_vars_list = [5, 6]
print('_'.join([str(e) for e in random_vars_list]))

perf_df_list = []
algos = ['SCM', 'DT', 'ICP+DT', 'ICP+SCM', 'ICSCM', 'RF']
for n_random_var in random_vars_list:
    print('n_random_var=', n_random_var)
    exec_time_1 = datetime.now()
    generated_df_list = []
    for repetition in range(10):
        df = compute_dataset(n_samples_per_env=10000, n_random_variables=n_random_var, noise_on_y=noise_on_y, noise_on_Xc=noise_on_Xc, random_seed=repetition)
        generated_df_list.append(df)
    feat_usage_df, perf_df = compute_features_usage_df(algos, generated_df_list, do_grid_search=True, param_grids=param_grids)
    perf_df['n_var'] = [n_random_var]*perf_df.shape[0]
    perf_df_list.append(perf_df)
    exec_time_2 = datetime.now()
    print('execution time=', (exec_time_2 - exec_time_1).total_seconds(), 'seconds')
    print(exec_time_2 - exec_time_1)


#old_perf_df = pd.read_csv('n_features_stats_df.csv')
#perf_df_list.append(old_perf_df)

big_perf_df = pd.concat(perf_df_list)
big_perf_df.to_csv('n_features_stats_df_' + '_'.join([str(e) for e in random_vars_list]) + '.csv', index=False)


# to plot the results :
"""
big_perf_df = pd.read_csv('n_features_stats_df.csv')
fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=False, figsize=(6, 7))
perf_df_runtime = big_perf_df[big_perf_df['type'] == 'fit time']
ax[0].set_title('runtime (in seconds to fit one model)')
sns.lineplot(data=perf_df_runtime, x='n_var', y='score', hue='algo', markers=True, dashes=False, ax=ax[0])
ax[0].set_yscale('log')

perf_df_causalscore = big_perf_df[big_perf_df['type'] == 'causal']
ax[1].set_title('causal score')
sns.lineplot(data=perf_df_causalscore, x='n_var', y='score', hue='algo', markers=True, dashes=False, ax=ax[1])

perf_df_predictive = big_perf_df[big_perf_df['type'] == 'test']
ax[2].set_title('predictive score (test)')
sns.lineplot(data=perf_df_predictive, x='n_var', y='score', hue='algo', markers=True, dashes=False, ax=ax[2])

#sns.lineplot(data=perf_df_perf, x='n_var', y='score', hue='algo', markers=True, dashes=False, ax=ax[2])
plt.xticks(sorted(list(set(big_perf_df['n_var']))))
plt.show()
"""
