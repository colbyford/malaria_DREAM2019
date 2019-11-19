# -*- coding: utf-8 -*-
"""
Malaria DREAM Challenge 2019
Script to Create AMLS Dataset
from SubChallenge 1 Data
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
Create Dataset from SubChallenge 1 Shaped Data
"""
from azureml.core.datastore import Datastore
from azureml.core.dataset import Dataset
# retrieve an existing datastore in the workspace by name
datastore = Datastore.get(ws, 'workspaceblobstore')

datastore_paths = [(datastore, 'SubCh1_TrainingData_Shaped.csv')]
sc1_ds = Dataset.Tabular.from_delimited_files(path=datastore_paths)

#%%
"""
Register the Dataset
"""

sc1_ds = sc1_ds.register(workspace=ws,
                         name='SubCh1_TrainingData_Shaped',
                         description='SubCh1 Shaped Training Data')