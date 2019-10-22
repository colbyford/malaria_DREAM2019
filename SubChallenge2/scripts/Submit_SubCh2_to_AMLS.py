# -*- coding: utf-8 -*-
"""
Malaria DREAM Challenge 2019
Script to Submit SubChallenge 2 Data to AMLS
By: Colby T. Ford, Ph.D.
"""
#%%
"""
Load in Libraries
"""
import json
import logging

import numpy as np
import pandas as pd

## pip install -U azureml-sdk --user
## pip install -U azureml.core --user
## pip install -U azureml.train.automl --user

## On macOS, you may have to install `brew install libomp` and then `pip install lightgbm` and run the following:
## import os
## os.environ['KMP_DUPLICATE_LIB_OK']='True'

import azureml.core
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
from azureml.train.automl import AutoMLConfig
from azureml.train.automl.run import AutoMLRun

#%%
"""
Get Information for the AMLS in Azure
"""
subscription_id = "0bb59590-d012-407d-a545-7513aae8c4a7" #you should be owner or contributor
resource_group = "DSBA6190-Class" #you should be owner or contributor
workspace_name = "dsba6190-amls" #your workspace name
workspace_region = "eastus2" #your regionsubscription_id = "" #You should be owner or contributor

#%%
"""
Setup the Workspace
"""
# Import the Workspace class and check the Azure ML SDK version.
from azureml.core import Workspace

ws = Workspace.create(name = workspace_name,
                      subscription_id = subscription_id,
                      resource_group = resource_group,
                      location = workspace_region,
                      exist_ok=True)
ws.get_details()

#%%
"""
Define the Experiment and Project
"""
#ws = Workspace.from_config()

# choose a name for experiment
experiment_name = 'automl-malariadream-sc2'
# project folder
project_folder = './aml_project/automl-malariadream-sc2'

experiment=Experiment(ws, experiment_name)

output = {}
output['SDK version'] = azureml.core.VERSION
output['Subscription ID'] = ws.subscription_id
output['Workspace'] = ws.name
output['Resource Group'] = ws.resource_group
output['Location'] = ws.location
output['Project Directory'] = project_folder
output['Experiment Name'] = experiment.name
pd.set_option('display.max_colwidth', -1)
outputDf = pd.DataFrame(data = output, index = [''])
outputDf.T

#%%
"""
Load in Data
"""
import pickle
X_train = pickle.load(open("../data/sc2_X_train.pkl", "rb"))
y_train = pickle.load(open("../data/sc2_y_train.pkl", "rb"))

#%%
"""
Configure AutoML
"""
automl_config = AutoMLConfig(task = 'classification',
                             name = experiment_name,
                             debug_log = 'automl_errors.log',
                             primary_metric = 'AUC_weighted',
                             iteration_timeout_minutes = 20,
                             iterations = 100,
                             max_cores_per_iteration = 14,
                             preprocess = True,
                             n_cross_validations = 10,
                             verbosity = logging.INFO,
                             model_explainability=True,
                             X = X_train, 
                             y = y_train,
                             path = project_folder)

#%%
"""
Submit to AutoML
"""
local_run = experiment.submit(automl_config, show_output = True)

#%%
"""
Retrieve Best Model and Save Locally
"""
best_run, fitted_model = local_run.get_output()
# print(best_run)
# print(fitted_model)

pickle.dump(fitted_model, open( "../model/amls_model_10-31-19/sc2_model.pkl", "wb" ) )

#%%
"""
Load in Model
"""
import pickle

fitted_model = pickle.load(open("../model/amls_model_10-31-19/sc2_model.pkl","rb"))

#%%
"""
Predict Test Data
"""
X_test = pickle.load(open("../data/sc2_X_test.pkl", "rb"))

y_predict = fitted_model.predict(X_test)
print(y_predict)

y_prob = fitted_model.predict_proba(X_test)
print(y_prob)

#%%
"""
Model Explanability
"""
from azureml.explain.model._internal.explanation_client import ExplanationClient

client = ExplanationClient.from_run_id(ws,
                                       experiment_name = "automl-malariadream-sc2",
                                       run_id = "AutoML_43d7399a-3f89-4728-b884-51be811a8e1a_98")

#client = ExplanationClient.from_run(best_run)
engineered_explanations = client.download_model_explanation(raw=False)
print(engineered_explanations.get_feature_importance_dict())

