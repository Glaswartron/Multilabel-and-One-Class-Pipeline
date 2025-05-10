"""
This module contains functions to build and test a classifier pipeline for multilabel or one-class classification problems.
"""

import numpy as np
import pandas as pd

import os
import sys
import argparse
import yaml
import pickle
import threading
from typing import Callable
import logging

from MLSMOTEOversampling import MLSMOTE
import Utility

sys.path.append("..")
from Common import hyperparameters

from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
import CustomSklearnUtilites
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report, confusion_matrix, get_scorer
from sklearn.base import BaseEstimator
from imblearn.over_sampling import SMOTE

import optuna

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


problem_types = ["multilabel", "oneclass"]

_multilabel_algorithms = {
    "XGBoost": (XGBClassifier, hyperparameters.get_params_xgb),
    "CatBoost": (CatBoostClassifier, hyperparameters.get_params_cat),
    "RandomForest": (RandomForestClassifier, hyperparameters.get_params_rf),
    "KNN": (KNeighborsClassifier, hyperparameters.get_params_knn)
}
_one_class_algorithms = {
    "OneClassSVM": (CustomSklearnUtilites.BinaryOneClassSVM, hyperparameters.get_params_oc_svm),
    "IsolationForest": (CustomSklearnUtilites.BinaryIsolationForest, hyperparameters.get_params_isoforest),
    "LocalOutlierFactor": (CustomSklearnUtilites.BinaryLocalOutlierFactor, hyperparameters.get_params_lof)
}


def build_classifier(train_data_path: str = None, labels: list[str] = None,
                     X_train_path: str = None, y_train_path: str = None,
                     problem_type: str = None, algorithm: str = None,
                     scale_data: bool = True, variance_threshold: bool | float = True, use_RFECV: bool = True, use_SMOTE: bool = False,
                     temp_results_path: str = None,
                     n_splits_cv: int = 3,
                     hpo_n_trials: int = 50,
                     scorer: Callable = None, optimization_direction: str = "maximize",
                     data_has_index: bool = True, 
                     n_jobs_optuna: int = 1, n_jobs_cv: int = 1):    
    """
    Build and optimize a classifier pipeline based on the provided parameters.

    Parameters
    -----------
    train_data_path : str, optional
        Path to the training data CSV file. Either this or X_train_path and y_train_path must be provided.
    labels : list of str, optional
        List of column names to be used as labels in the training data. Required if train_data_path is provided.
    X_train_path : str, optional
        Path to the CSV file containing training features. Either this or train_data_path and labels must be provided.
    y_train_path : str, optional
        Path to the CSV file containing training labels. Either this or train_data_path and labels must be provided.
    problem_type : str, optional
        Type of classification problem. Must be one of the predefined problem types.
    algorithm : str, optional
        Algorithm to be used for classification. Must be compatible with the problem type.
    scale_data : bool, default=True
        Whether to scale the data.
    variance_threshold : bool or float, default=True
        Whether to apply variance thresholding. If a float is provided, it is used as the threshold value.
    use_RFECV : bool, default=True
        Whether to use Recursive Feature Elimination with Cross-Validation (RFECV).
    use_SMOTE : bool, default=False
        Whether to use Synthetic Minority Over-sampling Technique (SMOTE) for balancing classes.
    temp_results_path : str, optional
        Path to store temporary results during optimization.
    n_splits_cv : int, default=3
        Number of splits for cross-validation.
    hpo_n_trials : int, default=50
        Number of trials for hyperparameter optimization.
    scorer : Callable, optional
        Scoring function to evaluate the predictions on the test set. See sklearn documentation for more information.
    optimization_direction : str, default="maximize"
        Direction of optimization, either "maximize" or "minimize".
    data_has_index : bool, default=True
        Whether the data files have an index column as the first column.
    n_jobs_optuna : int, default=1
        Number of parallel jobs for Optuna optimization.
    n_jobs_cv : int, default=1
        Number of parallel jobs for cross-validation.

    Returns
    --------
    classifier : sklearn.pipeline.Pipeline
        The optimized classifier pipeline.

    Raises
    -------
    ValueError
        If required parameters are not provided or if invalid values are given.
    """        
    
    
    if problem_type is None or problem_type not in problem_types:
        logging.error(f"Problem type must be provided as one of the following: {problem_types}")
        raise ValueError(f"Problem type must be provided as one of the following: {problem_types}")

    # Ensure that either train_data_path and labels are provided or X_train_path and y_train_path
    if ((not train_data_path) or labels is None or labels is []) and (not X_train_path or not y_train_path):
        logging.error("Either train_data_path and labels or X_train_path and y_train_path must be provided")
        raise ValueError("Either train_data_path and labels or X_train_path and y_train_path must be provided")
    
    if problem_type == "multilabel" and not algorithm in _multilabel_algorithms.keys():
        logging.error(f"Algorithm {algorithm} not supported for multi-label classification. Must be one of: {list(_multilabel_algorithms.keys())}")
        raise ValueError(f"Algorithm {algorithm} not supported for multi-label classification. Must be one of: {list(_multilabel_algorithms.keys())}")
    elif problem_type == "oneclass" and not algorithm in _one_class_algorithms.keys():
        logging.error(f"Algorithm {algorithm} not supported for one-class classification. Must be one of: {list(_one_class_algorithms.keys())}")
        raise ValueError(f"Algorithm {algorithm} not supported for one-class classification. Must be one of: {list(_one_class_algorithms.keys())}")

    if problem_type == "oneclass" and use_SMOTE:
        logging.warning("SMOTE is not supported for one-class classification, setting use_SMOTE to False")
        use_SMOTE = False

    # KNeighborsClassifier supports multi-label but doesnt give feature importances
    if algorithm == "KNN" and use_RFECV:
        logging.warning("RFECV is not supported for KNN, setting use_RFECV to False")
        use_RFECV = False

    # Default scoring
    if scorer is None:
        logging.info("No scorer function provided, using default scorer function: " +
                     "Macro F1 for multilabel classification, F1 for one-class classification")
        if problem_type == "multilabel":
            scorer = get_scorer("f1_macro")
        elif problem_type == "oneclass":
            scorer = get_scorer("f1")

    if problem_type == "multilabel" and use_RFECV and temp_results_path:
        logging.info("Will reuse classifier from RFECV inside optimization for later steps")
        if Utility.ensure_exists_dir(temp_results_path): logging.info(f"Created temp results directory at {temp_results_path}")
    
    # Load training data
    logging.info("Loading training data")
    X_train, y_train = Utility.load_data(train_data_path, labels, X_train_path, y_train_path, data_has_index)

    # Convert multi-label data to binary for one-class classification if necessary
    if problem_type == "oneclass" and y_train.shape[1] > 1:
        logging.info("Removing all other labels except for the first one (fault 1 = no, 0 = yes) for one-class classification and inverting it")
        # If the first label is 0 or any other label is one, the sample is faulty
        y_train = (1 - y_train[y_train.columns[0]]) | y_train[y_train.columns[1:]].any(axis=1) # bool-valued
        y_train = y_train.astype(int) # From here on: fault 1 = yes, 0 = no
    elif problem_type == "oneclass" and y_train.shape[1] == 1:
        y_train = y_train.iloc[:, 0] # Just convert to Series

    logging.info("Creating classifier pipeline")

    # For one-class classification, feature selection with linear-kernel one-class SVM can be done at the start
    if problem_type == "oneclass" and use_RFECV:
        logging.info("Doing RFECV before hyperparameter optimization for one-class classification")
        # Special kfold for one-class classification with only good samples in the training splits
        splits = CustomSklearnUtilites.split_with_one_class_KFold(X_train, y_train, n_splits=n_splits_cv, shuffle=True, random_state=42) 
        # Only linear kernel is supported for feature selection. Needs to be *Binary*OneClassSVM for RFECV.
        base_classifier = _make_base_classifier_pipeline("oneclass", CustomSklearnUtilites.BinaryOneClassSVM, {"kernel": "linear"},
                                                         scale_data, variance_threshold, use_SMOTE)
        # Perform RFECV
        importance_getter = "named_steps.classifier.coef_"
        rfecv = RFECV(base_classifier, cv=splits, n_jobs=n_jobs_cv, scoring=scorer, importance_getter=importance_getter)
        X_train = pd.DataFrame(rfecv.fit_transform(X_train, y_train), index=X_train.index) # X now has less features from here on
    else:
        rfecv = None

    # Optimize hyperparameters
    logging.info("Optimizing hyperparameters")
    study = optuna.create_study(direction=optimization_direction)
    objective = _make_optuna_objective(problem_type, X_train, y_train, algorithm, scorer,
                                       scale_data, variance_threshold, use_RFECV, use_SMOTE,
                                       temp_results_path, n_splits_cv, n_jobs_cv)
    study.optimize(objective, n_trials=hpo_n_trials, n_jobs=n_jobs_optuna)
    logging.info(f"Hyperparameter optimization finished with best validation value: {study.best_value}")

    logging.info("Building classifier with best hyperparameters")
    
    # Build pipeline/classifier with best hyperparameters
    if problem_type == "multilabel":
        return _get_or_refit_best_classifier_multilabel(study, X_train, y_train, algorithm, temp_results_path,
                                                        scale_data, variance_threshold, use_SMOTE, use_RFECV, 
                                                        scorer, n_splits_cv, n_jobs_cv)
    elif problem_type == "oneclass":
        return _refit_best_classifier_oneclass(study, X_train, y_train, algorithm, scale_data, variance_threshold, rfecv)


def _make_optuna_objective(problem_type, X_train, y_train, algorithm, scorer,
                           scale_data, variance_threshold, use_RFECV, use_SMOTE,
                           temp_results_path, n_splits_cv, n_jobs_cv):
    if problem_type == "multilabel":
        if temp_results_path is not None:
            if not os.path.exists(temp_results_path):
                os.makedirs(temp_results_path)
            file_operation_lock = threading.Lock()
        else:
            file_operation_lock = None
        return lambda trial: _optuna_objective_multilabel(trial, X_train, y_train, algorithm, scorer, scale_data, variance_threshold,
                                                          use_RFECV, use_SMOTE, temp_results_path, n_splits_cv, file_operation_lock, n_jobs_cv)
    elif problem_type == "oneclass":
        return lambda trial: _optuna_objective_oneclass(trial, X_train, y_train, algorithm, scorer, scale_data, variance_threshold, n_splits_cv)
    

def _make_base_classifier_pipeline(problem_type, model_class, model_hyperparameters, scale_data, variance_threshold, use_SMOTE):
    if problem_type == "oneclass" and model_class in [CustomSklearnUtilites.BinaryIsolationForest, CustomSklearnUtilites.BinaryLocalOutlierFactor]\
    and "contamination" not in model_hyperparameters:
        model_hyperparameters["contamination"] = sys.float_info.min # Just one class in the training data
    if problem_type == "multilabel" and model_class == CatBoostClassifier:
        model_hyperparameters["loss_function"] = "MultiLogloss" # Required for multilabel classification
        # No output during training
        model_hyperparameters["silent"] = True

    classifier_model = model_class(**model_hyperparameters)
    ''' 
    Add the other steps to the pipeline.
    No SMOTE for one-class data, should not even come here with use_SMOTE=True in one-class case.
    '''
    if not use_SMOTE or problem_type == "oneclass": 
        classifier_pipeline = Pipeline([
            ("classifier", classifier_model)
        ])
    else:
        smote_class = MLSMOTE if problem_type == "multilabel" else SMOTE
        classifier_pipeline = ImbPipeline([
            ('SMOTE', smote_class()), # Tied to estimator, used during every fold in RFECV and cross_val_score
            ('classifier', classifier_model)
        ])
    if variance_threshold:
        classifier_pipeline.steps.insert(0, ('variance_threshold', VarianceThreshold(variance_threshold)))
    if scale_data:
        classifier_pipeline.steps.insert(0, ('scaler', StandardScaler()))
    return classifier_pipeline
    

def _optuna_objective_multilabel(trial, X_train, y_train, algorithm, scorer,
                                 scale_data, variance_threshold, use_RFECV, use_SMOTE,
                                 temp_results_path, n_splits_cv, file_operation_lock, n_jobs_cv):

    classifier_class = _multilabel_algorithms[algorithm][0]
    classifier_hyperparameters = _multilabel_algorithms[algorithm][1](trial)
    base_classifier = _make_base_classifier_pipeline("multilabel", classifier_class, classifier_hyperparameters,
                                                     scale_data, variance_threshold, use_SMOTE)

    if not use_RFECV:
        # Normal training and cross-validation
        score = cross_val_score(base_classifier, X_train, y_train, cv=n_splits_cv, scoring=scorer, n_jobs=n_jobs_cv).mean()
        return score

    # Note that RFECV implements BaseEstimator, so it can be used as the classifier from here on
    importance_getter = "named_steps.classifier.feature_importances_"
    classifier = RFECV(base_classifier, cv=n_splits_cv, n_jobs=n_jobs_cv, scoring=scorer, importance_getter=importance_getter)
    classifier.fit(X_train, y_train)

    score = classifier.cv_results_["mean_test_score"][classifier.n_features_ - 1] # Only works if rfecv.step==1

    if temp_results_path is not None:
        with file_operation_lock: # Exclusive to avoid race conditions and lost updates with the other optuna threads
            # Get current best score
            saved_estimators = os.listdir(temp_results_path)
            if len(saved_estimators) != 0:
                scores = [float(estimator.split("_")[-1].split(".")[0]) for estimator in saved_estimators]
                best_score = np.max(scores)
            else:
                best_score = -1

            if score > best_score:
                # Delete old best estimator
                if best_score != -1:
                    best_estimator_path = saved_estimators[np.argmax(scores)]
                    os.remove(os.path.join(temp_results_path, best_estimator_path))
                # Save best estimator
                path = os.path.join(temp_results_path, f"temp_best_estimator_{score}.pkl")
                with open(path, 'wb') as f:
                    pickle.dump(classifier, f)
    
    trial.set_user_attr("n_features", classifier.n_features_)
    return score


def _optuna_objective_oneclass(trial, X_train, y_train, algorithm, scorer,
                               scale_data, variance_threshold, n_splits_cv):
    
    classifier_class = _one_class_algorithms[algorithm][0]
    classifier_hyperparameters = _one_class_algorithms[algorithm][1](trial)
    classifier = _make_base_classifier_pipeline("oneclass", classifier_class, classifier_hyperparameters, scale_data, variance_threshold, False)

    splits = CustomSklearnUtilites.split_with_one_class_KFold(X_train, y_train, n_splits=n_splits_cv, shuffle=True, random_state=42)

    return cross_val_score(classifier, X_train, y_train, cv=splits, scoring=scorer).mean()


def _get_or_refit_best_classifier_multilabel(study, X_train, y_train, algorithm, temp_results_path,
                                             scale_data, variance_threshold, use_SMOTE, use_RFECV, scorer,
                                             n_splits_cv, n_jobs_cv):

    if use_RFECV and temp_results_path is not None:
        # Use best classifier from during the optimization
        assert len(os.listdir(temp_results_path)) == 1, "Multiple saved models detected under temp_results_path" # Just in case there are race conditions or something
        best_estimator_path = os.listdir(temp_results_path)[0]
        with open(os.path.join(temp_results_path, best_estimator_path), 'rb') as f:
            classifier = pickle.load(f)

        # Delete temp results
        os.remove(os.path.join(temp_results_path, best_estimator_path))
        if len(os.listdir(temp_results_path)) == 0:
            os.rmdir(temp_results_path)
    else:
        # Refit classifier with best hyperparameters
        classifier_class = _multilabel_algorithms[algorithm][0]
        base_classifier = _make_base_classifier_pipeline("multilabel", classifier_class, study.best_params, scale_data, variance_threshold, use_SMOTE)
        if use_RFECV:
            importance_getter = "named_steps.classifier.feature_importances_"
            classifier = RFECV(base_classifier, cv=n_splits_cv, n_jobs=n_jobs_cv, scoring=scorer, importance_getter=importance_getter)
        else:
            classifier = base_classifier
        classifier.fit(X_train, y_train)

    return classifier


def _refit_best_classifier_oneclass(study, X_train, y_train, algorithm, scale_data, variance_threshold, rfecv):

    classifier = _one_class_algorithms[algorithm][0](**study.best_params)

    # RFECV is already fit, so the other steps are also fit beforehand. Final pipeline object can be used for predicting.

    # Remove all bad (positive class) samples from the training data for one-class classification
    X_train = X_train[y_train == 0]
    y_train = y_train[y_train == 0]

    if variance_threshold:
        variance_threshold = VarianceThreshold(variance_threshold)
        X_train = pd.DataFrame(variance_threshold.fit_transform(X_train), index=X_train.index)

    if scale_data:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)

    classifier.fit(X_train, y_train)

    pipeline_steps = []
    if rfecv:
        pipeline_steps.append(('use_RFECV', rfecv))
    if variance_threshold:
        pipeline_steps.append(('variance_threshold', variance_threshold))
    if scale_data:
        # Scaler has to come after feature selection
        pipeline_steps.append(('scaler', scaler))
    pipeline_steps.append(('classifier', classifier))
    classifier = Pipeline(pipeline_steps)

    return classifier


def test_classifier(classifier, problem_type: str,
                    test_data_path: str = None, labels: list[str] = None,
                    X_test_path: str = None, y_test_path: str = None,
                    features: list[str] = None,
                    data_has_index: bool = True):
    """
    Test a classifier on provided test data and return evaluation metrics.

    Parameters
    -----------
    classifier : BaseEstimator or AutoGluon predictor
        The classifier or pipeline to be tested.
    problem_type : str
        The type of problem. Must be one of the predefined problem types. If "oneclass", the function will handle the labels accordingly.
    test_data_path : str, optional
        Path to the CSV file containing the test data with features and labels.
    labels : list of str, optional
        List of column names in the test data that represent the labels.
    X_test_path : str, optional
        Path to the CSV file containing the test features.
    y_test_path : str, optional
        Path to the CSV file containing the test labels.
    features : list of str, optional
        List of feature names to be used for testing. A subset of columns in the test data. Should match the features the classifier was trained on.
    data_has_index : bool, default=True
        Indicates whether the CSV files have an index column as the first column.

    Returns
    --------
    clf_report : str
        Classification report as a string.
    conf_matrix : ndarray or None
        Confusion matrix if `problem_type` is "oneclass", otherwise None.

    Raises
    -------
    ValueError
        If neither (test_data_path and labels) nor (X_test_path and y_test_path) are provided.
    """
    
    # Ensure that either test_data_path and labels are provided or X_test_path and y_test_path
    if (test_data_path is None or labels is None) and (X_test_path is None or y_test_path is None):
        logging.error("Either test_data_path and labels or X_test_path and y_test_path must be provided")
        raise ValueError("Either test_data_path and labels or X_test_path and y_test_path must be provided")
    
    # Load test data
    logging.info("Loading test data")
    X_test, y_test = Utility.load_data(test_data_path, labels, X_test_path, y_test_path, data_has_index)

    if features is not None:
        X_test = X_test[features]

    if problem_type == "oneclass" and y_test.shape[1] > 1:
        logging.info("Removing all other labels except for the first one (fault yes/no) for one-class classification and inverting it")
        y_test = y_test[y_test.columns[0]]
        y_test = 1 - y_test

    # Evaluate classifier/pipeline
    y_pred = classifier.predict(X_test)
    clf_report = classification_report(y_test, y_pred)
    if problem_type == "oneclass":
        conf_matrix = confusion_matrix(y_test, y_pred)
    else:
        conf_matrix = None

    return clf_report, conf_matrix


def main():
    config, args = handle_args_and_config()
    problem_type = config["problem_type"]
    algorithm = config["algorithm"]
    results_path = config["results_path"]
    data_params = config["data"]
    pipeline_params = config["pipeline"]
    hpo_params = config["hpo"]
    pipeline_params["use_RFECV"] = pipeline_params["RFECV"] # Alias
    pipeline_params["use_SMOTE"] = pipeline_params["SMOTE"] # Alias

    # Init path stuff that was not provided to None
    for path in ["train_data_path", "X_train_path", "y_train_path", "test_data_path", "X_test_path", "y_test_path", "temp_results_path"]:
        if path not in data_params:
            data_params[path] = None
    if "labels" not in data_params:
        data_params["labels"] = None

    # Order important because logging needs results_path to exist
    dir_created = Utility.ensure_exists_dir(results_path)
    Utility.init_logging(args.logging_level, results_path)
    if dir_created: logging.info(f"Created results directory at {results_path}")

    # Write copy of the config file to the results directory for reproducibility
    results_path = config["results_path"]
    with open(os.path.join(results_path, "config.yaml"), 'w') as f:
        yaml.dump(config, f)

    scorer = get_scorer(hpo_params["metric"])

    classifier = build_classifier(problem_type=problem_type, algorithm=algorithm, scorer=scorer,
                                  train_data_path=data_params["train_data_path"], labels=data_params["labels"],
                                  X_train_path=data_params["X_train_path"], y_train_path=data_params["y_train_path"],
                                  scale_data=pipeline_params["scale_data"], variance_threshold=pipeline_params["variance_threshold"],
                                  use_RFECV=pipeline_params["use_RFECV"], use_SMOTE=pipeline_params["use_SMOTE"],
                                  n_splits_cv=pipeline_params["n_splits_cv"], hpo_n_trials=hpo_params["n_trials"],
                                  optimization_direction=hpo_params["optimization_direction"],
                                  temp_results_path=data_params["temp_results_path"], data_has_index=data_params["data_has_index"],
                                  n_jobs_optuna=hpo_params["n_jobs_optuna"], n_jobs_cv=pipeline_params["n_jobs_cv"])
    
    clf_report, conf_matrix = test_classifier(classifier, problem_type=problem_type, test_data_path=data_params["test_data_path"],
                                              labels=data_params["labels"], X_test_path=data_params["X_test_path"], y_test_path=data_params["y_test_path"],
                                              data_has_index=data_params["data_has_index"])
    logging.info("Classification report:\n" + clf_report)

    logging.info(f"Saving results to {results_path}")
    # Save classification report
    with open(os.path.join(results_path, "classification_report.txt"), 'w') as f:
        f.write(clf_report)
    # Save confusion matrix
    if conf_matrix is not None:
        with open(os.path.join(results_path, "confusion_matrix.txt"), 'w') as f:
            f.write(str(conf_matrix))
    # Save pipeline using pickle
    with open(os.path.join(results_path, f"classifier_{problem_type}.pkl"), 'wb') as f:
        pickle.dump(classifier, f)
    

def handle_args_and_config():
    parser = argparse.ArgumentParser(description='Build a multilabel or oneclass classifier')
    parser.add_argument("--config", type=str, help="Path to the yaml configuration file")
    parser.add_argument("--logging_level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()

    config = Utility.load_config(args.config)

    return config, args
    

if __name__ == '__main__':
    main()