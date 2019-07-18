# -*- coding: utf-8 -*-
"""
Malaria DREAM Challenge 2019
Script to Submit SubChallenge 1 Data to AMLS
By: Colby T. Ford, Ph.D.
"""
#%%
"""
Load in Libraries
"""
import json
import logging

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets

## pip install -U azureml-sdk --user
## pip install -U azureml.core --user
## pip install -U azureml.train.automl --user

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
experiment_name = 'automl-malariadream-sc1'
# project folder
project_folder = './aml_project/automl-malariadream-sc1'

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
X_train = pickle.load(open("../data/sc1_X_train.pkl", "rb"))
y_train = pickle.load(open("../data/sc1_y_train.pkl", "rb"))

#%%
"""
Configure AutoML
"""
automl_config = AutoMLConfig(task = 'regression',
                             name = experiment_name,
                             debug_log = 'automl_errors.log',
                             primary_metric = 'spearman_correlation', #'normalized_root_mean_squared_error'
                             iteration_timeout_minutes = 20,
                             iterations = 100,
                             max_cores_per_iteration = 4,
                             preprocess = True,
                             n_cross_validations = 10,
                             verbosity = logging.INFO,
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
Retrieve Best Model
"""
# best_run, fitted_model = local_run.get_output()
# print(best_run)
# print(fitted_model)

fitted_model = pickle.load(open("../model/amls_model_7-18-19/sc1_model.pkl","rb"))

#%%
"""
Predict Test Data
"""
X_test = pickle.load(open("../data/sc1_X_test.pkl", "rb"))

y_predict = fitted_model.predict(X_test)
print(y_predict)