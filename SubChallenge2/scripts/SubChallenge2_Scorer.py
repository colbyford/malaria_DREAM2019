# Databricks notebook source
# MAGIC %md
# MAGIC # Malaria DREAM Challenge 2019
# MAGIC ## Subchallenge 2 - Scorer
# MAGIC ------------------------------
# MAGIC ## Transform Data into Arrays

# COMMAND ----------

# DBTITLE 1,Data Prep - Load Data into Spark
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col

data = spark.read.format("csv") \
               .options(header = True, inferSchema = True) \
               .load("/mnt/malaria/SubCh2_TestData.csv") \
               .withColumnRenamed("Asexual_Stage", "Asexual_stage__hpi_")

pipeline = PipelineModel.load("/mnt/malaria/sc2/pipeline/")

data = pipeline.transform(data).select(col("label"), col("features"))

display(data)

# COMMAND ----------

# DBTITLE 1,Data Prep - Convert Spark DataFrame to Numpy Array
import numpy as np
## Whole Test Data
pdtest = data.toPandas()
testseries = pdtest['features'].apply(lambda x : np.array(x.toArray())).as_matrix().reshape(-1,1)
X_test = np.apply_along_axis(lambda x : x[0], 1, testseries)
# y_test = pdtest['label'].values.reshape(-1,1).ravel()

print(X_test)

# COMMAND ----------

import pickle
pickle.dump(X_test, open( "sc2_X_test.pkl", "wb" ) )
dbutils.fs.cp("file:/databricks/driver/sc2_X_test.pkl", "/mnt/malaria/sc2/arraydata")

display(dbutils.fs.ls("/mnt/malaria/sc2/arraydata"))

# COMMAND ----------

# MAGIC %md
# MAGIC --------------------
# MAGIC ## Azure Machine Learning Service - AutoML

# COMMAND ----------

# DBTITLE 1,Load in Libraries
import json
import logging

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets

## pip install azureml-sdk
import azureml.core
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
from azureml.train.automl import AutoMLConfig
from azureml.train.automl.run import AutoMLRun

# COMMAND ----------

# DBTITLE 1,Get Information for the AMLS in Azure
subscription_id = "0bb59590-d012-407d-a545-7513aae8c4a7" #you should be owner or contributor
resource_group = "DSBA6190-Class" #you should be owner or contributor
workspace_name = "dsba6190-amls" #your workspace name
workspace_region = "eastus2" #your regionsubscription_id = "" #You should be owner or contributor

# COMMAND ----------

# DBTITLE 1,Setup the Workspace
# Import the Workspace class and check the Azure ML SDK version.
from azureml.core import Workspace

ws = Workspace.create(name = workspace_name,
                      subscription_id = subscription_id,
                      resource_group = resource_group, 
                      location = workspace_region,                      
                      exist_ok=True)
ws.get_details()

# COMMAND ----------

# DBTITLE 1,Define the Experiment and Project
#ws = Workspace.from_config()

# choose a name for experiment
experiment_name = 'automl-malariadream-sc2'
# project folder
project_folder = './sample_projects/automl-malariadream-sc2'

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

# COMMAND ----------

# DBTITLE 1,Get Best Model
best_run, fitted_model = local_run.get_output()
print(best_run)
print(fitted_model)

# COMMAND ----------

# DBTITLE 1,Predict Test Data
y_hat = fitted_model.predict(X_test.values)