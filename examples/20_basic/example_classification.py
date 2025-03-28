# -*- encoding: utf-8 -*-
"""
==============
Classification
==============

The following example shows how to fit a simple classification model with
*auto-sklearn*.
"""
import sklearn.datasets
import sklearn.metrics

import autosklearn.classification


############################################################################
# Data Loading
# ============

X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(X, y, random_state=1)

results = dict()
for x in [None, 'LightGBM']:
    ############################################################################
    # Build and fit a classifier
    # ==========================

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=1800,
        per_run_time_limit=30,
        memory_limit=32768,
        surrogate_model=x
    )
    automl.fit(X_train, y_train, dataset_name='breast_cancer')

    ############################################################################
    # View the models found by auto-sklearn
    # =====================================

    print(automl.leaderboard())

    ############################################################################
    # Print the final ensemble constructed by auto-sklearn
    # ====================================================

    print(automl.show_models())

    ###########################################################################
    # Get the Score of the final ensemble
    # ===================================

    predictions = automl.predict(X_test)
    print("Accuracy score:", sklearn.metrics.accuracy_score(y_test, predictions))

    results[f'Leaderboard {x}'] = automl.leaderboard()
    results[f'Models {x}'] = automl.show_models()
    results[f'Accuracy [{x}'] = sklearn.metrics.accuracy_score(y_test, predictions)
