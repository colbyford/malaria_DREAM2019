# Databricks notebook source
# MAGIC %md
# MAGIC # Malaria DREAM Challenge 2019
# MAGIC ## Subchallenge 1
# MAGIC ------------------------------
# MAGIC ### AutoML - Azure Machine Learning Service

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
experiment_name = 'automl-malariadream-sc1'
# project folder
project_folder = './sample_projects/automl-malariadream-sc1'

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

# DBTITLE 1,Data Prep - Load Data into Spark
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col

data = spark.read.format("csv") \
               .options(header = True, inferSchema = True) \
               .load("/mnt/malaria/SubCh1_TrainingData.csv")

pipeline = PipelineModel.load("/mnt/malaria/sc1/pipeline/")

data = pipeline.transform(data).select(col("label"), col("features"))
train, test = data.randomSplit([0.75, 0.25], 1337)

# test = spark.read.format("csv") \
#                .options(header = True, inferSchema = True) \
#                .load("/mnt/malaria/SubCh1_TestData.csv")
# test = pipeline.transform(test).select(col("label"), col("features"))

display(train)

# COMMAND ----------

# DBTITLE 1,Data Prep - Convert Spark DataFrame to Numpy Array
## Training Data
pdtrain = train.toPandas()
trainseries = pdtrain['features'].apply(lambda x : np.array(x.toArray())).as_matrix().reshape(-1,1)
X_train = np.apply_along_axis(lambda x : x[0], 1, trainseries)
y_train = pdtrain['label'].values.reshape(-1,1).ravel()

## Test Data
pdtest = test.toPandas()
testseries = pdtest['features'].apply(lambda x : np.array(x.toArray())).as_matrix().reshape(-1,1)
X_test = np.apply_along_axis(lambda x : x[0], 1, testseries)
y_test = pdtest['label'].values.reshape(-1,1).ravel()

print(y_test)

# COMMAND ----------

# DBTITLE 1,Configure AutoML
automl_config = AutoMLConfig(task = 'regression',
                             name = experiment_name,
                             debug_log = 'automl_errors.log',
                             primary_metric = 'normalized_root_mean_squared_error',
                             iteration_timeout_minutes = 20,
                             iterations = 100,
                             preprocess = True,
                             n_cross_validations = 5,
                             verbosity = logging.INFO,
                             X = X_train, 
                             y = y_train,
                             path = project_folder)

# primary_metric = 'normalized_root_mean_squared_error',

# COMMAND ----------

# DBTITLE 1,Submit to AutoML
local_run = experiment.submit(automl_config, show_output = True)

# COMMAND ----------

# MAGIC %md
# MAGIC -----------------------------
# MAGIC ### Train with Spark MLlib

# COMMAND ----------

# DBTITLE 1,Linear Regression
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator

# Create initial LinearRegression model
lr = LinearRegression(labelCol="label", featuresCol="features")


# Create ParamGrid for Cross Validation
lrparamGrid = (ParamGridBuilder()
             #.addGrid(lr.regParam, [0.001, 0.01, 0.1, 0.5, 1.0, 2.0])
               .addGrid(lr.regParam, [0.01, 0.1, 0.5])
             #.addGrid(lr.elasticNetParam, [0.0, 0.25, 0.5, 0.75, 1.0])
               .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
             #.addGrid(lr.maxIter, [1, 5, 10, 20, 50])
               .addGrid(lr.maxIter, [1, 5, 10])
             .build())

# Evaluate model
lrevaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label", metricName="rmse")

# Create 5-fold CrossValidator
lrcv = CrossValidator(estimator = lr,
                    estimatorParamMaps = lrparamGrid,
                    evaluator = lrevaluator,
                    numFolds = 5)

# Run cross validations
lrcvModel = lrcv.fit(train)
print(lrcvModel)

# Get Model Summary Statistics
lrcvSummary = lrcvModel.bestModel.summary
print("Coefficient Standard Errors: " + str(lrcvSummary.coefficientStandardErrors))
print("P Values: " + str(lrcvSummary.pValues)) # Last element is the intercept

# Use test set here so we can measure the accuracy of our model on new data
lrpredictions = lrcvModel.transform(test)

# cvModel uses the best model found from the Cross Validation
# Evaluate best model
print('RMSE:', lrevaluator.evaluate(lrpredictions))

# COMMAND ----------

from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator

# Create an initial RandomForest model.
rf = RandomForestRegressor(labelCol="label", featuresCol="features")

# Evaluate model
rfevaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label", metricName="rmse")

# Create ParamGrid for Cross Validation
rfparamGrid = (ParamGridBuilder()
             #.addGrid(rf.maxDepth, [2, 5, 10, 20, 30])
               .addGrid(rf.maxDepth, [2, 5, 10])
             #.addGrid(rf.maxBins, [10, 20, 40, 80, 100])
               .addGrid(rf.maxBins, [5, 10, 20])
             #.addGrid(rf.numTrees, [5, 20, 50, 100, 500])
               .addGrid(rf.numTrees, [5, 20, 50])
             .build())

# Create 5-fold CrossValidator
rfcv = CrossValidator(estimator = rf,
                      estimatorParamMaps = rfparamGrid,
                      evaluator = rfevaluator,
                      numFolds = 5)

# Run cross validations.
rfcvModel = rfcv.fit(train)
print(rfcvModel)

# Use test set here so we can measure the accuracy of our model on new data
rfpredictions = rfcvModel.transform(test)

# cvModel uses the best model found from the Cross Validation
# Evaluate best model
print('RMSE:', rfevaluator.evaluate(rfpredictions))

# COMMAND ----------

# DBTITLE 1,Save Models
lrcvModel.save("/mnt/malaria/sc1/trainedmodels/lr")
rfcvModel.save("/mnt/malaria/sc1/trainedmodels/rf")

display(dbutils.fs.ls("/mnt/trainedmodels/"))

# COMMAND ----------

# MAGIC %md
# MAGIC ----------------------
# MAGIC ### Score Submission Data

# COMMAND ----------

subdata = spark.read.format("csv") \
               .options(header = True, inferSchema = True) \
               .load("/mnt/malaria/SubCh1_TestData.csv")

pipeline = PipelineModel.load("/mnt/malaria/sc1/pipeline/")

subdata = pipeline.transform(subdata).select(col("label"), col("features"))

output = lrcvModel.bestModel.transform(subdata)

display(output)