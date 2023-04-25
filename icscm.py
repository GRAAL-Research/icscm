"""
    FlorenceSetCoveringMachine1 -- A casuality oriented version of the Set Covering Machine in Python
    Copyright (C) 2022 Thibaud Godon, Florence Clerc, Alexandre Drouin

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

import math
import logging
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
    check_random_state,
)
from warnings import warn

from pingouin import chi2_independence
import warnings


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


class BaseInvariantCausalSCM1(BaseEstimator, ClassifierMixin):
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
        self.threshold = threshold
        self.resample_rules = resample_rules
        self.stopping_method = stopping_method
        self.random_state = random_state

    def get_params(self, deep=True):
        return {
            "p": self.p,
            "model_type": self.model_type,
            "max_rules": self.max_rules,
            "random_state": self.random_state,
        }

    def set_params(self, **parameters):
        for parameter, value in iteritems(parameters):
            setattr(self, parameter, value)
        return self

    def fit(self, X, y, n_env, tiebreaker=None, iteration_callback=None, **fit_params):
        """
        Fit a SCM model.

        Parameters:
        -----------
        X: array-like, shape=[n_examples, n_features]
            The feature of the input examples.
        y : array-like, shape = [n_samples]
            The labels of the input examples.
        tiebreaker: function(model_type, feature_idx, thresholds, rule_type)
            A function that takes in the model type and information about the
            equivalent rules and outputs the index of the rule to use. The lists
            respectively contain the feature i
            #print('residuals.shape[1], len(all_possible_rules)', residuals.shape[1], len(all_possible_rules))
            assert residuals.shape[1] == len(all_possible_rules)

            for i in range(residuals.shape[1]):
                res = residuals[:, i]
                res_e_df = pd.DataFrame({'res': res, 'e': env_of_remaining_examples})
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    #print('confusion_matrix', confusion_matrix(res_e_df['res'], res_e_df['e']))
                p_vals.append(p_value)
                utility_true_negndices, thresholds and type
            corresponding of the equivalent rules. If None, the rule that most
            decreases the training error is selected. Note: the model type is
            provided because the rules that are added to disjunction models
            correspond to the inverse of the rules that are handled during
            training. Handle this case with care.
        iteration_callback: function(model)
            A function that is called each time a rule is added to the model.

        Returns:
        --------
        self: object
            Returns self.

        """
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
        logging.debug("Parsing additional fit parameters")
        utility_function_additional_args = {}
        if fit_params is not None:
            for key, value in iteritems(fit_params):
                if key[:9] == "utility__":
                    utility_function_additional_args[key[9:]] = value

        # Validate the input data
        logging.debug("Validating the input data")
        # X, y = check_X_y(X, y)
        # X = np.asarray(X, dtype=np.double)
        self.classes_, y, total_n_ex_by_class = np.unique(
            y, return_inverse=True, return_counts=True
        )
        if len(self.classes_) != 2:
            raise ValueError("y must contain two unique classes.")
        logging.debug(
            "The data contains {0:d} examples. Negative class is {1!s} (n: {2:d}) and positive class is {3!s} (n: {4:d}).".format(
                len(y),
                self.classes_[0],
                total_n_ex_by_class[0],
                self.classes_[1],
                total_n_ex_by_class[1],
            )
        )

        # Invert the classes if we are learning a disjunction
        # logging.debug("Preprocessing example labels")
        # pos_ex_idx, neg_ex_idx = self._get_example_idx_by_class(y)
        # y = np.zeros(len(y), dtype=np.int)
        # y[pos_ex_idx] = 1
        # y[neg_ex_idx] = 0

        # Presort all the features
        ##logging.debug("Presorting all features")
        ##X_argsort_by_feature_T = np.argsort(X, axis=0).T.copy()
        # print('X', X)
        # print('y', y)
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
        logging.debug("Initializing empty model")
        self.model_ = ConjunctionModel()
        logging.debug("Training start")
        ones = np.ones(len(y))
        remaining_N = ones - y  # remaining negative examples
        remaining_P = y

        env_of_remaining_examples = np.array([int(np.where(x[:n_env])[0]) for x in X])

        # Extract features only
        X_original_with_env = (
            X.copy()
        )  # useful for prediction for feature importance computation
        X = X[:, n_env:]
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
        n_rules_with_indep_pos_residuals = [len(all_possible_rules)]
        # self.threshold = 0.005
        # print('self.threshold', self.threshold)

        while not (stopping_criterion):
            # while (len(remaining_y) - sum(remaining_y)) > 0 and len(self.model_) < self.max_rules:
            error_by_rule = residuals.sum(axis=0)
            # print('error_by_rule', error_by_rule)
            n_rules_with_indep_neg_residuals.append(0)
            n_rules_with_indep_pos_residuals.append(0)
            # print('residuals.shape', residuals.shape)
            # print('best rule  by accuracy :', all_possible_rules[error_by_rule.argmin()])
            # We seek rules with residuals that are invariant to the environment, so high p-values
            p_vals_neg_leafs, p_vals_pos_leafs = [], []
            utilities = []
            scores_of_rules = []
            # print('residuals.shape[1], len(all_possible_rules)', residuals.shape[1], len(all_possible_rules))
            assert residuals.shape[1] == len(all_possible_rules)
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
                y_e_df = pd.DataFrame(
                    {
                        "y": remaining_y,
                        "e": env_of_remaining_examples,
                        "rule_pred": big_pred_matrix[:, i],
                    }
                )
                # print('res_e_df.shape', res_e_df.shape)
                neg_leaf_y_e_df = y_e_df[y_e_df["rule_pred"] == 0]
                pos_leaf_y_e_df = y_e_df[y_e_df["rule_pred"] == 1]
                # print('neg_res_e_df.shape', neg_res_e_df.shape)
                if len(neg_leaf_y_e_df) == 0:
                    p_value_neg_leaf = 1
                elif len(pos_leaf_y_e_df) == 0:
                    p_value_pos_leaf = 1
                else:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        # p value computed on the residuals of the negative leaf of the rule
                        p_value_neg_leaf = chi2_independence(
                            data=neg_leaf_y_e_df, x="y", y="e"
                        )[-1]["pval"][0]
                        p_value_pos_leaf = chi2_independence(
                            data=pos_leaf_y_e_df, x="y", y="e"
                        )[-1]["pval"][0]
                p_vals_neg_leafs.append(p_value_neg_leaf)
                p_vals_pos_leafs.append(p_value_pos_leaf)
                n_rules_with_indep_neg_residuals[-1] = n_rules_with_indep_neg_residuals[
                    -1
                ] + int(p_value_neg_leaf > self.threshold)
                n_rules_with_indep_pos_residuals[-1] = n_rules_with_indep_pos_residuals[
                    -1
                ] + int(p_value_pos_leaf > self.threshold)
                score_of_rule = int(p_value_neg_leaf > self.threshold) * utility
                scores_of_rules.append(score_of_rule)
                # print('rule = feature {} {:2} {}     p_value = {:3f} |{:10}|     utility = {:5d} |{:10}|      score = {:5d}'.format(rule_feat_id, '>' if rule_kind == 'greater' else '<=', rule_threshold, p_value, '#'*int(10*p_value), int(utility), '+'*int(10*ponderated_utility), int(score_of_rule)))
            p_vals_neg_leafs = np.array(p_vals_neg_leafs)
            p_vals_pos_leafs = np.array(p_vals_pos_leafs)
            best_rule_id = np.array(scores_of_rules).argmax()
            # print('np.array(scores_of_rules).argmax()', np.array(scores_of_rules).argmax())
            best_rule_score = scores_of_rules[best_rule_id]
            best_rule_feat_id, best_rule_threshold, best_rule_kind = all_possible_rules[
                best_rule_id
            ]
            # update possible rules:
            # print('p_values', p_vals)

            mask = np.zeros(big_pred_matrix.shape, dtype=bool)
            updated_all_possible_rules = []
            assert (
                len(p_vals_neg_leafs)
                == len(p_vals_pos_leafs)
                == len(all_possible_rules)
            )
            # print(' --- updating possible rules ---')
            for i, rule in enumerate(all_possible_rules):
                # print('rule', rule)
                if rule[0] != best_rule_feat_id:
                    if self.resample_rules:
                        updated_all_possible_rules.append(rule)
                    elif p_vals_neg_leafs[i] > self.threshold:
                        updated_all_possible_rules.append(rule)
            # print('len(updated_all_possible_rules)', len(updated_all_possible_rules))
            ##if (len(updated_all_possible_rules) == 0):
            ##	break
            # columns_to_keep = np.array([rule[0] for rule in updated_all_possible_rules])
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
                break
            samples_to_keep = np.array(predictions_of_selected_rule).astype(bool)
            for i in range(big_pred_matrix.shape[0]):
                if samples_to_keep[i]:
                    mask[i] = columns_to_keep
            new_dimensions = (sum(samples_to_keep), sum(columns_to_keep))
            # print('new_dimensions', new_dimensions)
            updated_big_pred_matrix = big_pred_matrix[mask].reshape(new_dimensions)
            updated_residuals = residuals[mask].reshape(new_dimensions)
            remaining_y = remaining_y[samples_to_keep]
            env_of_remaining_examples = env_of_remaining_examples[samples_to_keep]
            big_pred_matrix = updated_big_pred_matrix
            residuals = updated_residuals
            all_possible_rules = updated_all_possible_rules

            global_best_rule_feat_id = best_rule_feat_id + n_env
            stump = DecisionStump(
                feature_idx=global_best_rule_feat_id,
                threshold=best_rule_threshold,
                kind=best_rule_kind,
            )

            # print('FNre', FNre)
            # print('sum(FNre)', sum(FNre))
            # print('TNre', TNre)
            # print('sum(TNre)', sum(TNre))
            # if sum(TNre) == sum(FNre) == 0:
            #    break

            # print("The best rule has utility {}".format(best_utility))
            # print("The best rule has score {}".format(best_rule_score))
            self._add_attribute_to_model(stump)

            stopping_criterion = False
            if len(self.model_) >= self.max_rules:
                print(
                    "len(self.model_) >= self.max_rules",
                    len(self.model_),
                    self.max_rules,
                    "stopping",
                )
                stopping_criterion = True
            elif len(remaining_y) == 0:
                print("len(remaining_y) == 0", len(remaining_y), "stopping")
                stopping_criterion = True
            elif len(all_possible_rules) == 0:
                print(
                    "len(all_possible_rules) == 0", len(all_possible_rules), "stopping"
                )
                stopping_criterion = True
            else:
                if self.stopping_method == "no_more_negatives":
                    print(
                        "self.stopping_method == no_more_negatives",
                        (len(remaining_y) == sum(remaining_y)),
                        "stopping if True",
                    )
                    stopping_criterion = len(remaining_y) == sum(
                        remaining_y
                    )  # only positive examples remaining
                elif self.stopping_method == "independance_y_e":
                    assert len(remaining_y) == len(env_of_remaining_examples)
                    y_e_df = pd.DataFrame(
                        {"remaining_y": remaining_y, "e": env_of_remaining_examples}
                    )
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        p_value_stopping = chi2_independence(
                            data=y_e_df, x="remaining_y", y="e"
                        )[-1]["pval"][0]
                    stopping_criterion = p_value_stopping > self.threshold
                    print(
                        "(p_value_stopping > self.threshold)",
                        (p_value_stopping > self.threshold),
                        "stopping if True",
                    )
                else:
                    raise ValueError(
                        "unexpected stopping_criterion", self.stopping_method
                    )

            logging.debug(
                "Discarding all examples that the rule classifies as negative"
            )

            # print("There are {} examples remaining ({} negatives)".format(sum(remaining_N) + sum(remaining_P), sum(remaining_N)))
            logging.debug(
                "There are {} examples remaining ({} negatives)".format(
                    sum(remaining_N) + sum(remaining_P), sum(remaining_N)
                )
            )

            iteration_callback(self.model_)

            # print('#####################################################################################')
            # print('len(self.model_)', len(self.model_))
            # print('self.model_', self.model_)
            # print('#####################################################################################')

        logging.debug("Training completed")
        # print(' *** trained FSCM1 : ')n_rules_with_indep_neg_residuals
        # print(rule)
        # print('    rule # : feat_id {} is {} than {}'.format(rule))

        self.n_rules_with_indep_neg_residuals = n_rules_with_indep_neg_residuals

        logging.debug("Calculating rule importances")
        # Definition: how often each rule outputs a value that causes the value of the model to be final
        final_outcome = 0 if self.model_type == "conjunction" else 1
        total_outcome = (
            self.model_.predict(X_original_with_env) == final_outcome
        ).sum()  # n times the model outputs the final outcome
        self.rule_importances_ = np.array(
            [
                (r.classify(X_original_with_env) == final_outcome).sum() / total_outcome
                for r in self.model_.rules
            ]
        )  # contribution of each rule
        logging.debug("Done.")

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
        check_is_fitted(self, ["model_", "rule_importances_", "classes_"])
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
        check_is_fitted(self, ["model_", "rule_importances_", "classes_"])
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
        check_is_fitted(self, ["model_", "rule_importances_", "classes_"])
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


class InvariantCausalSCM1(BaseInvariantCausalSCM1):
    """
    A Set Covering Machine classifier

    [1]_ Marchand, M., & Shawe-Taylor, J. (2002). The set covering machine.
    Journal of Machine Learning Research, 3(Dec), 723-746.

    Parameters:
    -----------
    p: float
        The trade-off parameter for the utility function (suggestion: use values >= 1).
    model_type: str, default="conjunction"
        The model type (conjunction or disjunction).
    max_rules: int, default=10
        The maximum number of rules in the model.
    random_state: int, np.random.RandomState or None, default=None
        The random state.

    """

    def __init__(
        self,
        p=1.0,
        model_type=str("conjunction"),
        max_rules=10,
        resample_rules=False,
        threshold=0.05,
        stopping_method="independance_y_e",
        random_state=None,
    ):
        super(InvariantCausalSCM1, self).__init__(
            p=p,
            model_type=model_type,
            max_rules=max_rules,
            resample_rules=False,
            threshold=threshold,
            stopping_method=stopping_method,
            random_state=random_state,
        )

    def _get_best_utility_rules(self, X, y, X_argsort_by_feature_T, example_idx):
        return find_max_utility(self.p, X, y, X_argsort_by_feature_T, example_idx)
