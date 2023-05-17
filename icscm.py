"""
    Invariant Causal Set Covering Machine -- A invariance-oriented version of the Set Covering Machine in Python
    Copyright (C) 2023 Thibaud Godon, Alexandre Drouin

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
from __future__ import print_function, division, absolute_import, unicode_literals

from six import iteritems

import logging
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
    check_random_state,
)
from warnings import warn

#from pingouin import chi2_independence
from scipy.stats.contingency import expected_freq
from scipy.stats import power_divergence
import warnings

from io import StringIO


def _class_to_string(instance):
    """
    Returns a string representation of the public attributes of a class.

    Parameters:
    -----------
    instance: object
        An instance of any class.

    Returns:
    --------
    string_rep: string
        A string representation of the class and its public attributes.

    Notes:
    -----
    Private attributes must be marked with a leading underscore.
    """
    return (
        instance.__class__.__name__
        + "("
        + ",".join(
            [
                str(k) + "=" + str(v)
                for k, v in iteritems(instance.__dict__)
                if str(k[0]) != "_"
            ]
        )
        + ")"
    )


class BaseModel(object):
    def __init__(self):
        self.rules = []
        super(BaseModel, self).__init__()

    def add(self, rule):
        self.rules.append(rule)

    def predict(self, X):
        raise NotImplementedError()

    def predict_proba(self, X):
        raise NotImplementedError()

    def remove(self, index):
        del self.rules[index]

    @property
    def example_dependencies(self):
        return [d for ba in self.rules for d in ba.example_dependencies]

    @property
    def type(self):
        raise NotImplementedError()

    def _to_string(self, separator=" "):
        return separator.join([str(a) for a in self.rules])

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __iter__(self):
        for ba in self.rules:
            yield ba

    def __len__(self):
        return len(self.rules)

    def __str__(self):
        return self._to_string()


class ConjunctionModel(BaseModel):
    def predict(self, X):
        predictions = np.ones(X.shape[0], bool)
        for a in self.rules:
            predictions = np.logical_and(predictions, a.classify(X))
        return predictions.astype(np.uint8)

    @property
    def type(self):
        return "conjunction"

    def __str__(self):
        return self._to_string(separator=" and ")


class BaseRule(object):
    """
    A rule mixin class

    """

    def __init__(self):
        super(BaseRule, self).__init__()

    def classify(self, X):
        """
        Classifies a set of examples using the rule.

        Parameters:
        -----------
        X: array-like, shape=(n_examples, n_features), dtype=np.float
            The feature vectors of examples to classify.

        Returns:
        --------
        classifications: array-like, shape=(n_examples,), dtype=bool
            The outcome of the rule (True or False) for each example.

        """
        raise NotImplementedError()

    def inverse(self):
        """
        Creates a rule that is the opposite of the current rule (self).

        Returns:
        --------
        inverse: BaseRule
            A rule that is the inverse of self.

        """
        raise NotImplementedError()

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __str__(self):
        return _class_to_string(self)


class DecisionStump(BaseRule):
    """
    A decision stump is a rule that applies a threshold to the value of some feature

    Parameters:
    -----------
    feature_idx: uint
        The index of the feature
    threshold: float
        The threshold at which the outcome of the rule changes
    kind: str, default="greater"
        The case in which the rule returns 1, either "greater" or "less_equal".

    """

    def __init__(self, feature_idx, threshold, kind="greater"):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.kind = kind
        super(DecisionStump, self).__init__()

    def classify(self, X):
        """
        Classifies a set of examples using the decision stump.

        Parameters:
        -----------
        X: array-like, shape=(n_examples, n_features), dtype=np.float
            The feature vectors of examples to classify.

        Returns:
        --------
        classifications: array-like, shape=(n_examples,), dtype=bool
            The outcome of the rule (True or False) for each example.

        """
        if self.kind == "greater":
            c = X[:, self.feature_idx] > self.threshold
        else:
            c = X[:, self.feature_idx] <= self.threshold
        return c

    def inverse(self):
        """
        Creates a rule that is the opposite of the current rule (self).

        Returns:
        --------
        inverse: BaseRule
            A rule that is the inverse of self.

        """
        return DecisionStump(
            feature_idx=self.feature_idx,
            threshold=self.threshold,
            kind="greater" if self.kind == "less_equal" else "less_equal",
        )

    def __str__(self):
        return "X[{0:d}] {1!s} {2:.3f}".format(
            self.feature_idx, ">" if self.kind == "greater" else "<=", self.threshold
        )


class InvariantCausalSCM(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        p=1.0,
        model_type="conjunction",
        max_rules=10,
        resample_rules=False,
        threshold=0.05,
        stopping_method="no_more_negatives",
        random_state=None,
    ):
        self.p = p
        self.model_type = model_type
        if model_type != "conjunction":
            raise ValueError(
                "wrong model_type: {}, only conjunction is supported".format(model_type)
            )
        self.max_rules = max_rules
        self.resample_rules = resample_rules
        self.threshold = threshold
        self.stopping_method = stopping_method
        self.random_state = random_state

    def get_params(self, deep=True):
        return {
            "p": self.p,
            "model_type": self.model_type,
            "max_rules": self.max_rules,
            "resample_rules": self.resample_rules,
            "threshold": self.threshold,
            "stopping_method": self.stopping_method,
            "random_state": self.random_state,
        }

    def set_params(self, **parameters):
        for parameter, value in iteritems(parameters):
            setattr(self, parameter, value)
        return self

    def fit(self, X, y, tiebreaker=None, iteration_callback=None, **fit_params):
        """
        Fit a SCM model.

        Parameters:
        -----------
        X: array-like, shape=[n_examples, n_features]
            The first columns contains the environment id of the example.
            The features of the input examples.
        y : array-like, shape = [n_samples]
            The labels of the input examples.

        Returns:
        --------
        self: object
            Returns self.

        """
        self.stream = StringIO()
        #self.handler = logging.StreamHandler(self.stream)
        #self.log = logging.getLogger('mylogger')
        #self.log.setLevel(logging.INFO)
        #for handler in self.log.handlers: 
        #    self.log.removeHandler(handler)
        #self.log.addHandler(self.handler)
        #def testLog(self):
        #self.assertEqual(self.stream.getvalue(), 'test')
        #def tearDown(self):
        #self.log.removeHandler(self.handler)
        #self.handler.close()

        random_state = check_random_state(self.random_state)

        if self.model_type == "conjunction":
            self._add_attribute_to_model = self._append_conjunction_model
            self._get_example_idx_by_class = self._get_example_idx_by_class_conjunction
        elif self.model_type == "disjunction":
            self._add_attribute_to_model = self._append_disjunction_model
            self._get_example_idx_by_class = self._get_example_idx_by_class_disjunction
        else:
            raise ValueError("Unsupported model type.")

        # Initialize callbacks
        if iteration_callback is None:
            iteration_callback = lambda x: None

        # Parse additional fit parameters
        self.stream.write('\n')
        self.stream.write("Parsing additional fit parameters")
        utility_function_additional_args = {}
        if fit_params is not None:
            for key, value in iteritems(fit_params):
                if key[:9] == "utility__":
                    utility_function_additional_args[key[9:]] = value

        # Validate the input data
        self.stream.write('\n')
        self.stream.write("Validating the input data")
        # X, y = check_X_y(X, y)
        # X = np.asarray(X, dtype=np.double)
        self.classes_, y, total_n_ex_by_class = np.unique(
            y, return_inverse=True, return_counts=True
        )
        if len(self.classes_) != 2:
            raise ValueError("y must contain two unique classes.")
        self.stream.write('\n')
        self.stream.write(
            "The data contains {0:d} examples. Negative class is {1!s} (n: {2:d}) and positive class is {3!s} (n: {4:d}).".format(
                len(y),
                self.classes_[0],
                total_n_ex_by_class[0],
                self.classes_[1],
                total_n_ex_by_class[1],
            )
        )

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        elif isinstance(X, np.ndarray):
            pass
        else:
            raise ValueError("unexpected type for X:", type(X))

        if isinstance(y, list):
            y = np.array(y)
        elif isinstance(y, np.ndarray):
            pass
        else:
            raise ValueError("unexpected type for y:", type(X))

        # Create an empty model
        self.stream.write('\n')
        self.stream.write("Initializing empty model")
        self.model_ = ConjunctionModel()
        self.stream.write('\n')
        self.stream.write("Training start")
        ones = np.ones(len(y))
        #remaining_N = ones - y  # remaining negative examples
        #remaining_P = y

        # first column of X: environment
        env_of_remaining_examples = X[:, 0]
        # print("env_of_remaining_examples", env_of_remaining_examples)

        # Extract features only
        X_original_with_env = (
            X.copy()
        )  # useful for prediction for feature importance computation
        X = X[:, 1:]
        # print('Extract features only, X.shape = ', X.shape)
        remaining_y = y

        all_possible_rules = []
        for feat_id in range(X.shape[1]):
            for threshold in list(set(X[:, feat_id])):
                for kind in ["greater", "less_equal"]:
                    all_possible_rules.append((feat_id, threshold, kind))

        # print('Number of examples per environment', np.bincount(env_of_remaining_examples))
        # print('Number of examples per label', np.bincount(remaining_y))

        big_pred_matrix = np.zeros((X.shape[0], len(all_possible_rules)), dtype=int)
        # print('X[1]', X[1])
        for sample_id in range(X.shape[0]):
            x = X[sample_id]
            for rule_id in range(len(all_possible_rules)):
                # print(all_possible_rules[rule_id])
                rule_feat_id, rule_threshold, rule_kind = all_possible_rules[rule_id]
                sample_feat_value = x[rule_feat_id]
                if rule_kind == "greater":
                    pred = 1 if (sample_feat_value > rule_threshold) else 0
                elif rule_kind == "less_equal":
                    pred = 1 if (sample_feat_value <= rule_threshold) else 0
                else:
                    raise ValueError("unexpected rule kind:", rule_kind)
                big_pred_matrix[sample_id, rule_id] = pred

        # Calculate residuals
        residuals = (big_pred_matrix != remaining_y[:, None]).astype(int)
        stopping_criterion = False
        n_rules_with_indep_neg_residuals = [len(all_possible_rules)]
        # self.threshold = 0.005
        # print('self.threshold', self.threshold)

        while not (stopping_criterion):
            # while (len(remaining_y) - sum(remaining_y)) > 0 and len(self.model_) < self.max_rules:
            #error_by_rule = residuals.sum(axis=0)
            # print('error_by_rule', error_by_rule)
            n_rules_with_indep_neg_residuals.append(0)
            # print('residuals.shape', residuals.shape)
            # print('best rule  by accuracy :', all_possible_rules[error_by_rule.argmin()])
            # We seek rules with residuals that are invariant to the environment, so high p-values
            p_vals_neg_leafs = []
            utilities = []
            scores_of_rules = []
            # print('residuals.shape[1], len(all_possible_rules)', residuals.shape[1], len(all_possible_rules))
            assert residuals.shape[1] == len(all_possible_rules)
            self.stream.write('\n')
            self.stream.write(f'len(all_possible_rules) = {len(all_possible_rules)}')
            y_e_df = pd.DataFrame(
                {
                    "y": remaining_y,
                    "e": env_of_remaining_examples,
                }
            )
            for i in range(residuals.shape[1]):
                res = residuals[:, i]  # erreurs de la regle
                utility_true_negatives = np.logical_not(
                    np.logical_or(res, remaining_y)
                ).astype(int)
                utility_false_negatives = np.logical_and(res, remaining_y).astype(int)
                utility = sum(utility_true_negatives) - self.p * sum(
                    utility_false_negatives
                )
                rule_feat_id, rule_threshold, rule_kind = all_possible_rules[i]
                utilities.append(utility)
                y_e_df["rule_pred"] = big_pred_matrix[:, i]
                # print('res_e_df.shape', res_e_df.shape)
                neg_leaf_y_e_df = y_e_df[y_e_df["rule_pred"] == 0]
                # print('neg_res_e_df.shape', neg_res_e_df.shape)
                if len(neg_leaf_y_e_df) == 0:
                    p_value_neg_leaf = 1
                else:
                    observed = pd.crosstab(neg_leaf_y_e_df['y'], neg_leaf_y_e_df['e'])
                    if observed.size == 0:
                        raise ValueError("No data; observed has size 0.")
                    
                    expected = pd.DataFrame(expected_freq(observed), index=observed.index, columns=observed.columns)

                    ### All count frequencies should be at least 5
                    #for df, name in zip([observed, expected], ["observed", "expected"]):
                    #    if (df < 5).any(axis=None):
                    #        warnings.warn("Low count on {} frequencies.".format(name))
                    #        print('expected', expected)
                    #        print('observed', observed)

                    dof = float(expected.size - sum(expected.shape) + expected.ndim - 1)

                    if dof == 1:
                        # Adjust `observed` according to Yates' correction for continuity.
                        observed = observed + 0.5 * np.sign(expected - observed)

                    ddof = observed.size - 1 - dof
                    n = len(remaining_y)
                    stats = []
                    lambda_ = 1.0
                    if dof == 0:
                        chi2, p_value_neg_leaf, cramer, power = 0.0, 1.0, np.nan, np.nan
                    else:
                        chi2, p_value_neg_leaf = power_divergence(observed, expected, ddof=ddof, axis=None, lambda_=lambda_)
                p_vals_neg_leafs.append(p_value_neg_leaf)
                n_rules_with_indep_neg_residuals[-1] = n_rules_with_indep_neg_residuals[
                    -1
                ] + int(p_value_neg_leaf > self.threshold)
                score_of_rule = int(p_value_neg_leaf > self.threshold) * utility
                scores_of_rules.append(score_of_rule)
                self.stream.write('\n')
                self.stream.write('rule : feature {} {:2} {}     p_value_neg_leaf = {:3f} |{:10}|     utility = {:5d}       score = {:5d}'.format(rule_feat_id, '>' if rule_kind == 'greater' else '<=', rule_threshold, p_value_neg_leaf, '#'*int(10*p_value_neg_leaf), int(utility), int(score_of_rule)))
            p_vals_neg_leafs = np.array(p_vals_neg_leafs)
            best_rule_id = np.array(scores_of_rules).argmax()
            best_rule_score = scores_of_rules[best_rule_id]
            best_rule_feat_id, best_rule_threshold, best_rule_kind = all_possible_rules[
                best_rule_id
            ]
            self.stream.write('\n')
            self.stream.write('selected best rule : "feature {} {:2} {}"'.format(best_rule_feat_id, '>' if best_rule_kind == 'greater' else '<=', best_rule_threshold))

            mask = np.zeros(big_pred_matrix.shape, dtype=bool)
            #updated_all_possible_rules = []
            #assert len(p_vals_neg_leafs) == len(all_possible_rules)
            ## print(' --- updating possible rules ---')
            #for i, rule in enumerate(all_possible_rules):
            #    # print('rule', rule)
            #    if rule[0] != best_rule_feat_id:
            #        if self.resample_rules:
            #            updated_all_possible_rules.append(rule)
            #        elif p_vals_neg_leafs[i] > self.threshold:
            #            updated_all_possible_rules.append(rule)
            # print('len(updated_all_possible_rules)', len(updated_all_possible_rules))
            updated_all_possible_rules = all_possible_rules.copy()
            ##if (len(updated_all_possible_rules) == 0):
            ##	break
            columns_to_keep = np.array(
                [(rule in updated_all_possible_rules) for rule in all_possible_rules]
            )
            predictions_of_selected_rule = big_pred_matrix[:, best_rule_id]
            classified_neg_examples = [
                (r == p == 0)
                for r, p in zip(
                    residuals[:, best_rule_id], predictions_of_selected_rule
                )
            ]
            # print('sum(predictions_of_selected_rule), sum(residuals[:, best_rule_id]), len(residuals[:, best_rule_id]), classified_neg_examples', sum(predictions_of_selected_rule), sum(residuals[:, best_rule_id]), len(residuals[:, best_rule_id]), sum(classified_neg_examples))
            # if (sum(residuals[:, best_rule_id]) == len(residuals[:, best_rule_id])) or (sum(classified_neg_examples) == 0):
            if sum(classified_neg_examples) == 0:
                self.stream.write('\n')
                self.stream.write('no more negative examples: breaking the while loop')
                break
            samples_to_keep = np.array(predictions_of_selected_rule).astype(bool)
            for i in range(big_pred_matrix.shape[0]):
                if samples_to_keep[i]:
                    mask[i] = columns_to_keep
            new_dimensions = (sum(samples_to_keep), sum(columns_to_keep))
            self.stream.write('\n')
            self.stream.write('new_dimensions = {}'.format(new_dimensions))
            updated_big_pred_matrix = big_pred_matrix[mask].reshape(new_dimensions)
            updated_residuals = residuals[mask].reshape(new_dimensions)
            remaining_y = remaining_y[samples_to_keep]
            env_of_remaining_examples = env_of_remaining_examples[samples_to_keep]
            big_pred_matrix = updated_big_pred_matrix
            residuals = updated_residuals
            all_possible_rules = updated_all_possible_rules

            global_best_rule_feat_id = best_rule_feat_id + 1
            stump = DecisionStump(
                feature_idx=global_best_rule_feat_id,
                threshold=best_rule_threshold,
                kind=best_rule_kind,
            )

            self.stream.write('\n')
            self.stream.write("The best rule has score {}".format(best_rule_score))
            self._add_attribute_to_model(stump)

            stopping_criterion = False
            self.stream.write('\n')
            self.stream.write('evaluation of stopping conditions : ')
            self.stream.write('\n')
            self.stream.write(f"len(self.model_) >= self.max_rules : {len(self.model_) >= self.max_rules} (len(self.model_) = {len(self.model_)}, self.max_rules = {self.max_rules}")
            self.stream.write('\n')
            self.stream.write(f'len(remaining_y) == 0 : {len(remaining_y) == 0}')
            self.stream.write('\n')
            self.stream.write(f'len(all_possible_rules) == 0 : {len(all_possible_rules) == 0}')
            
            if len(self.model_) >= self.max_rules:
                self.stream.write('\n')
                self.stream.write(f"len(self.model_) >= self.max_rules {len(self.model_)}, {self.max_rules} stopping")
                stopping_criterion = True
            elif len(remaining_y) == 0:
                self.stream.write('\n')
                self.stream.write(f"len(remaining_y) == 0 : {len(remaining_y)} stopping")
                stopping_criterion = True
            elif len(all_possible_rules) == 0:
                self.stream.write('\n')
                self.stream.write(
                   f"len(all_possible_rules) == 0 : {len(all_possible_rules)} stopping")
                stopping_criterion = True
            else:
                if self.stopping_method == "no_more_negatives":
                    self.stream.write('\n')
                    self.stream.write(f"self.stopping_method == no_more_negatives {(len(remaining_y) == sum(remaining_y))} stopping if True")
                    stopping_criterion = len(remaining_y) == sum(
                        remaining_y
                    )  # only positive examples remaining
                elif self.stopping_method == "independance_y_e":
                    self.stream.write('\n')
                    self.stream.write(f'independance_y_e')
                    assert len(remaining_y) == len(env_of_remaining_examples)
                    y_e_df = pd.DataFrame(
                        {"remaining_y": remaining_y, "e": env_of_remaining_examples}
                    )


                    observed = pd.crosstab(y_e_df['remaining_y'], y_e_df['e'])
                    if observed.size == 0:
                        raise ValueError("No data; observed has size 0.")
                    
                    expected = pd.DataFrame(expected_freq(observed), index=observed.index, columns=observed.columns)

                    ## All count frequencies should be at least 5
                    #for df, name in zip([observed, expected], ["observed", "expected"]):
                    #    if (df < 5).any(axis=None):
                    #        warnings.warn("Low count on {} frequencies.".format(name))

                    dof = float(expected.size - sum(expected.shape) + expected.ndim - 1)

                    if dof == 1:
                        # Adjust `observed` according to Yates' correction for continuity.
                        observed = observed + 0.5 * np.sign(expected - observed)

                    ddof = observed.size - 1 - dof
                    n = len(remaining_y)
                    stats = []
                    lambda_ = 1.0
                    if dof == 0:
                        chi2, p_value_stopping, cramer, power = 0.0, 1.0, np.nan, np.nan
                    else:
                        chi2, p_value_stopping = power_divergence(observed, expected, ddof=ddof, axis=None, lambda_=lambda_)
                        dof_cramer = min(expected.shape) - 1
                    stopping_criterion = p_value_stopping > self.threshold
                    self.stream.write('\n')
                    self.stream.write(f"(p_value_stopping = {p_value_stopping} | self.threshold = {self.threshold}")
                    self.stream.write('\n')
                    self.stream.write(f"(p_value_stopping > self.threshold) {(p_value_stopping > self.threshold)} (stopping if True)")
                else:
                    raise ValueError(
                        "unexpected stopping_criterion", self.stopping_method
                    )

            self.stream.write('\n')
            self.stream.write(
                "Discarding all examples that the rule classifies as negative"
            )

            self.stream.write('\n')

            iteration_callback(self.model_)

            # print('#####################################################################################')
            # print('len(self.model_)', len(self.model_))
            # print('self.model_', self.model_)
            # print('#####################################################################################')

        self.stream.write('\n')
        self.stream.write("Training completed")
        # print(' *** trained FSCM1 : ')n_rules_with_indep_neg_residuals
        # print(rule)
        # print('    rule # : feat_id {} is {} than {}'.format(rule))

        self.n_rules_with_indep_neg_residuals = n_rules_with_indep_neg_residuals

        self.stream.write('\n')
        self.stream.write("Calculating rule importances")
        # Definition: how often each rule outputs a value that causes the value of the model to be final
        final_outcome = 0 if self.model_type == "conjunction" else 1
        total_outcome = (
            self.model_.predict(X_original_with_env) == final_outcome
        ).sum()  # n times the model outputs the final outcome
        self.rule_importances = np.array(
            [
                (r.classify(X_original_with_env) == final_outcome).sum() / total_outcome
                for r in self.model_.rules
            ]
        )  # contribution of each rule
        self.stream.write('\n')
        self.stream.write("Done.")

        #print('log_stream.getvalue()', self.log_stream.getvalue())
        #self.handler.flush()

        return self

    def predict(self, X):
        """
        Predict class

        Parameters:
        -----------
        X: array-like, shape=[n_examples, n_features]
            The feature of the input examples.

        Returns:
        --------
        predictions: numpy_array, shape=[n_examples]
            The predicted class for each example.

        """
        check_is_fitted(self, ["model_", "rule_importances", "classes_"])
        X = check_array(X)
        return self.classes_[self.model_.predict(X)]

    def predict_proba(self, X):
        """
        Predict class probabilities

        Parameters:
        -----------
        X: array-like, shape=(n_examples, n_features)
            The feature of the input examples.

        Returns:
        --------
        p : array of shape = [n_examples, 2]
            The class probabilities for each example. Classes are ordered by lexicographic order.

        """
        warn(
            "SetCoveringMachines do not support probabilistic predictions. The returned values will be zero or one.",
            RuntimeWarning,
        )
        check_is_fitted(self, ["model_", "rule_importances", "classes_"])
        X = check_array(X)
        pos_proba = self.classes_[self.model_.predict(X)]
        neg_proba = 1.0 - pos_proba
        return np.hstack((neg_proba.reshape(-1, 1), pos_proba.reshape(-1, 1)))

    def score(self, X, y):
        """
        Predict classes of examples and measure accuracy

        Parameters:
        -----------
        X: array-like, shape=(n_examples, n_features)
            The feature of the input examples.
        y : array-like, shape = [n_samples]
            The labels of the input examples.

        Returns:
        --------
        accuracy: float
            The proportion of correctly classified examples.

        """
        check_is_fitted(self, ["model_", "rule_importances", "classes_"])
        X, y = check_X_y(X, y)
        return accuracy_score(y_true=y, y_pred=self.predict(X))

    def _append_conjunction_model(self, new_rule):
        self.model_.add(new_rule)
        logging.debug("Attribute added to the model: " + str(new_rule))
        return new_rule

    def _append_disjunction_model(self, new_rule):
        new_rule = new_rule.inverse()
        self.model_.add(new_rule)
        logging.debug("Attribute added to the model: " + str(new_rule))
        return new_rule

    def _get_example_idx_by_class_conjunction(self, y):
        positive_example_idx = np.where(y == 1)[0]
        negative_example_idx = np.where(y == 0)[0]
        return positive_example_idx, negative_example_idx

    def _get_example_idx_by_class_disjunction(self, y):
        positive_example_idx = np.where(y == 0)[0]
        negative_example_idx = np.where(y == 1)[0]
        return positive_example_idx, negative_example_idx

    def __str__(self):
        return _class_to_string(self)
