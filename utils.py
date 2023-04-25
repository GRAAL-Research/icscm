# compute_features_usage_df
import numpy as np
import pandas as pd


from pyscm import SetCoveringMachineClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from icscm import InvariantCausalSCM

from gsq import ci_tests
from itertools import chain, combinations
import warnings

def compute_features_usage_df(algos, df1, do_grid_search=False, param_grids=None):
    df3 = df1.copy()
    y = df3['Y'].values
    del df3['Y']
    X = df3.values
    features_names = list(df3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)
    features_usage_df = pd.DataFrame(columns=algos, index=features_names)
    perf_df = pd.DataFrame(columns=['algo', 'score', 'metric', 'type'])
    row_i = 0
    ##############
    # define data for home-made algos
    # add columns for each environment
    X_train_df_for_home_algos = pd.DataFrame(X_train, columns=features_names)
    X_test_df_for_home_algos = pd.DataFrame(X_test, columns=features_names)
    full_env_list = list(X_train_df_for_home_algos['E']) + list(X_test_df_for_home_algos['E'])
    n_env = len(set(full_env_list))
    id = 0
    for e in set(full_env_list):
        print(e)
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
        if algo == 'ICP':
            pass
        else:
            if algo == 'SCM':
                model = SetCoveringMachineClassifier(random_state=11)
            elif algo == 'SCM 1rule':
                model = SetCoveringMachineClassifier(max_rules=1, random_state=11)
            elif algo == 'DT':
                model = DecisionTreeClassifier(random_state=11)
            elif algo == 'ICSCM':
                model = InvariantCausalSCM(threshold=0.05, random_state=11)
            elif algo == 'ICSCM_stop_nomoreneg':
                model = InvariantCausalSCM(threshold=0.05, stopping_method='no_more_negatives', random_state=11)
            elif algo == 'ICSCM_stop_indepEY':
                model = InvariantCausalSCM(threshold=0.05, stopping_method='independance_y_e', random_state=11)
            else:
                raise ValueError('unknown algo', algo)
            if algo in ['robustSCM', 'residualSCM', 'residualSCM resample', 'residualSCMneg', 'FSCM1', 'FSCM1min', 'FSCM2', 'ICSCM', 'ICSCM_stop_nomoreneg', 'ICSCM_stop_indepEY']:
                model.fit(X_train_df_for_home_algos, y_train, n_env)
                test_pred = model.predict(X_test_df_for_home_algos)
                train_pred = model.predict(X_train_df_for_home_algos)
            else:
                # grid search for best parameters
                if do_grid_search:
                    hyperparameters = param_grids[algo]
                    grid = GridSearchCV(
                        estimator=model,
                        param_grid=hyperparameters,
                        verbose=1,
                        n_jobs=1,
                    )
                    grid_result = grid.fit(X_train, y_train)
                    tuned_hyperparameters = grid_result.best_params_
                    #cv_results_df = pd.DataFrame(grid_result.cv_results_)  # convert GS results to a pandas dataframe
                    model.set_params(**tuned_hyperparameters)  # set best params
                    model.set_params(random_state=11)  # set random state
                model.fit(X_train, y_train)
                test_pred = model.predict(X_test)
                train_pred = model.predict(X_train)
            perf_df.loc[row_i] = [algo, accuracy_score(y_test, test_pred), 'accuracy', 'test']
            row_i += 1
            perf_df.loc[row_i] = [algo, accuracy_score(y_train, train_pred), 'accuracy', 'train']
            row_i += 1
            if 'SCM' in algo:
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
                elif algo in ['residualSCM', 'residualSCM resample', 'residualSCMneg', 'FSCM1', 'FSCM1min', 'FSCM2', 'ICSCM', 'ICSCM_stop_nomoreneg', 'ICSCM_stop_indepEY']:
                    rule_importances = model.rule_importances_
                    for i in range(len(model.model_.rules)):
                        if model.model_.rules[i].feature_idx < n_env:
                            # one the env features is used here
                            features_used[features_names.index('E')] = rule_importances[i]
                        else:
                            features_used[model.model_.rules[i].feature_idx - n_env + 1] = rule_importances[i]
            elif algo in ['RF', 'DT']:
                features_used = model.feature_importances_
            elif algo == 'regression':
                positive_coeffs = [abs(coeff) for coeff in model.coef_[0]/sum(model.coef_[0])]
                features_used = positive_coeffs/sum(positive_coeffs)
            elif 'SVM' in algo:
                features_used = np.abs(model.coef_.flatten())
                features_used = features_used/sum(features_used)
            else:
                raise Exception('algo not implemented')
        features_usage_df[algo] = features_used
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

def compute_indep_test(df1, env_var, y_var):
    other_variables = [k for k in range(0, len(list(df1.columns)))]
    other_variables.remove(env_var)
    other_variables.remove(y_var)
    sets = list(chain.from_iterable(combinations(other_variables, r) for r in range(len(other_variables)+1)))
    sets_that_creates_indep = []
    for s in sets:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            p_value = ci_tests.ci_test_dis(df1.values, env_var, y_var, list(s))
        if p_value > 0.05:
            sets_that_creates_indep.append(s)
    return sets_that_creates_indep