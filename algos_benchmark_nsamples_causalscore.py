import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# compute_features_usage_df
import importlib
from pyscm import SetCoveringMachineClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

################## reload home algos if updated
import icscm
import icpscm
import icpdt
importlib.reload(icscm)
importlib.reload(icpscm)
importlib.reload(icpdt)
from icscm import InvariantCausalSCM
from icpscm import InvariantCausalPredictionSetCoveringMachine
from icpdt import InvariantCausalPredictionDecisionTree



# *********************** DEFINE DATA ***********************

#                                E=0  E=1
probas_table_for_a1 = np.array([[0.9, 0.5],  # Xa1=0
                                [0.1, 0.5]]) # Xa1=1

#                                E=0  E=1
probas_table_for_a2 = np.array([[0.5, 0.7],  # Xa2=0
                                [0.5, 0.3]]) # Xa2=1

noise_on_y = 0.05
noise_on_Xc = 0.05

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

def compute_dataset(n_samples_per_env, noise_on_y, noise_on_Xc, random_seed=11):
    random.seed(random_seed)
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


param_grids = {
    'ICSCM': {'p': [0.1, 0.5, 0.75, 1.0, 2.5, 5, 10], 'model_type': ['conjunction'], 'stopping_method': ['independance_y_e']},
    'ICSCM_nomoreneg': {'p': [0.1, 0.5, 0.75, 1.0, 2.5, 5, 10], 'model_type': ['conjunction'], 'stopping_method': ['no_more_negatives']},
    'ICSCM_indepEY': {'p': [0.1, 0.5, 0.75, 1.0, 2.5, 5, 10], 'model_type': ['conjunction'], 'stopping_method': ['independance_y_e']},
    'SCM': {'p': [0.1, 0.5, 0.75, 1.0, 2.5, 5, 10], 'model_type': ['conjunction']}, 
    'ICP+SCM': {'p': [0.1, 0.5, 0.75, 1.0, 2.5, 5, 10], 'model_type': ['conjunction']}, 
    'DT':     {'min_samples_split' : [2, 0.01, 0.05, 0.1, 0.3], 'max_depth' : [1, 2, 3, 4, 5, 10]},
    'ICP+DT': {'min_samples_split' : [2, 0.01, 0.05, 0.1, 0.3], 'max_depth' : [1, 2, 3, 4, 5, 10]}, 
}

def compute_features_usage_df(algos, generated_df_list, do_grid_search=False, param_grids=None):
    features_usage_df_list = []
    perf_df = pd.DataFrame(columns=['algo', 'score', 'metric', 'type', 'split'])
    row_i = 0
    for repetition in range(len(generated_df_list)):
        df2 = generated_df_list[repetition].copy()
        y = df2['Y'].values
        del df2['Y']
        X = df2.values
        features_names = list(df2)
        #print('features_names', features_names)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11+repetition)
        features_usage_df = pd.DataFrame(columns=algos, index=features_names)
        ##############
        # define data for home-made algos
        # add columns for each environment
        X_train_df_for_home_algos = pd.DataFrame(X_train, columns=features_names)
        X_test_df_for_home_algos = pd.DataFrame(X_test, columns=features_names)
        full_env_list = list(X_train_df_for_home_algos['E']) + list(X_test_df_for_home_algos['E'])
        n_env = len(set(full_env_list))
        id = 0
        for e in set(full_env_list):
            env_col_train = (X_train_df_for_home_algos['E'] == e).astype(int)
            env_col_test = (X_test_df_for_home_algos['E'] == e).astype(int)
            X_train_df_for_home_algos.insert(loc=0, column='E'+str(id), value=env_col_train)
            X_test_df_for_home_algos.insert(loc=0, column='E'+str(id), value=env_col_test)
            id += 1
        del X_train_df_for_home_algos['E']
        del X_test_df_for_home_algos['E']
        ##############
        for algo in algos:
            print(algo)
            if algo == 'SCM':
                model = SetCoveringMachineClassifier(random_state=11)
            elif algo == 'SCM 1rule':
                model = SetCoveringMachineClassifier(max_rules=1, random_state=11)
            elif algo == 'DT':
                model = DecisionTreeClassifier(random_state=11)
            elif algo == 'ICSCM':
                model = InvariantCausalSCM(threshold=0.05, random_state=11)
            elif algo == 'ICSCM_nomoreneg':
                model = InvariantCausalSCM(threshold=0.05, stopping_method='no_more_negatives', random_state=11)
            elif algo == 'ICSCM_indepEY':
                model = InvariantCausalSCM(threshold=0.05, stopping_method='independance_y_e', random_state=11)
            elif algo == 'ICP+SCM':
                model = InvariantCausalPredictionSetCoveringMachine(threshold=0.05, random_state=11)
            elif algo == 'ICP+DT':
                model = InvariantCausalPredictionDecisionTree(threshold=0.05, random_state=11)
            else:
                raise ValueError('unknown algo', algo)
            if algo in ['robustSCM', 'residualSCM', 'residualSCM resample', 'residualSCMneg', 'FSCM1', 'FSCM1min', 'FSCM2']:
                model.fit(X_train_df_for_home_algos, y_train, n_env)
                test_pred = model.predict(X_test_df_for_home_algos)
                train_pred = model.predict(X_train_df_for_home_algos)
            else:
                # grid search for best parameters
                if do_grid_search and (algo in param_grids):
                    hyperparameters = param_grids[algo]
                    grid = GridSearchCV(
                        estimator=model,
                        param_grid=hyperparameters,
                        verbose=1,
                        n_jobs=8,
                    )
                    grid_result = grid.fit(X_train, y_train)
                    tuned_hyperparameters = grid_result.best_params_
                    print('tuned_hyperparameters', tuned_hyperparameters)
                    #cv_results_df = pd.DataFrame(grid_result.cv_results_)  # convert GS results to a pandas dataframe
                    model.set_params(**tuned_hyperparameters)  # set best params
                    model.set_params(random_state=11)  # set random state
                model.fit(X_train, y_train)
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
                elif algo == 'robustSCM':
                    for i in range(len(model.estim.model_.rules)):
                        #feat_name = features_names[model.estim.model_.rules[i].feature_idx]
                        if model.estim.model_.rules[i].feature_idx < n_env:
                            # one the env features is used here
                            features_used[features_names.index('E')] = model.estim.rule_importances[i]
                        else:
                            features_used[model.estim.model_.rules[i].feature_idx - n_env + 1] = model.estim.rule_importances[i]
                elif algo in ['residualSCM', 'residualSCM resample', 'residualSCMneg', 'FSCM1', 'FSCM1min', 'FSCM2']:
                    rule_importances = model.rule_importances_
                    for i in range(len(model.model_.rules)):
                        if model.model_.rules[i].feature_idx < n_env:
                            # one the env features is used here
                            features_used[features_names.index('E')] = rule_importances[i]
                        else:
                            features_used[model.model_.rules[i].feature_idx - n_env + 1] = rule_importances[i]                    
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
        features_usage_df['feature'] = features_usage_df.index
        features_usage_df['split'] = [repetition]*features_usage_df.shape[0]
        features_usage_df_list.append(features_usage_df)
    features_usage_df = pd.concat(features_usage_df_list, axis=0)
    return features_usage_df, perf_df

algos = ['ICP+SCM', 'ICSCM_nomoreneg', 'ICSCM_indepEY', 'SCM']
n_samples_list = [20000]
try:
    algos_scores_df = pd.read_csv('n_samples_causal_scores_df.csv')
    print('loaded algos_scores_df')
    row_i = algos_scores_df.shape[0]
except:
    algos_scores_df = pd.DataFrame(columns=['n_samples', 'algo', 'causal score', 'split'])
    row_i = 0
true_causal_features_idx = ['Xa1', 'Xa2']
for n_samples in n_samples_list:
    print('n_samples = {}'.format(n_samples))
    df_list = []
    for i in range(5):
        df = compute_dataset(n_samples, noise_on_y=0.1, noise_on_Xc=0.05, random_seed=i)
        df_list.append(df)
    feat_usage_df, perf_df = compute_features_usage_df(algos, df_list, do_grid_search=True, param_grids=param_grids)
    for split in feat_usage_df['split']:
        loc_feat_usage_df = feat_usage_df[feat_usage_df['split'] == split]
        for a in algos:
            features_usage = loc_feat_usage_df[a]
            features_used = features_usage[features_usage > 0]
            causal_score = int(set(list(features_used.index)) == set(true_causal_features_idx))
            #if causal_score == 1:
            #    print(list(features_used.index), 'causal_score = {}'.format(causal_score))
            algos_scores_df.loc[row_i] = [n_samples, a, causal_score, split]
            row_i += 1

algos_scores_df.to_csv('n_samples_causal_scores_df.csv', index=False)

# to plot the results :
"""
algos_scores_df = pd.read_csv('n_samples_causal_scores_df.csv')
sns.lineplot(data=algos_scores_df, x='n_samples', y='causal score', hue='algo', markers=True, dashes=False)
plt.title('Causal score')
plt.xticks(sorted(list(set(algos_scores_df['n_samples']))))
#plt.xscale('log')
plt.show()
"""