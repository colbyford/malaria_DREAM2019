# Databricks notebook source
# MAGIC %md
# MAGIC # Malaria DREAM Challenge 2019
# MAGIC ## Subchallenge 1
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

# COMMAND ----------

local_run = experiment.submit(automl_config, show_output = True)

# COMMAND ----------

local_run

# COMMAND ----------

# MAGIC %md Retrieve the Best Model
# MAGIC Below we select the best pipeline from our iterations. The get_output method on automl_classifier returns the best run and the fitted model for the last invocation. Overloads on get_output allow you to retrieve the best run and fitted model for any logged metric or for a particular iteration.

# COMMAND ----------

best_run, fitted_model = local_run.get_output()

# COMMAND ----------

print(best_run)
print(fitted_model)
print(type(fitted_model))

# COMMAND ----------

# MAGIC %md Register the Fitted Model for Deployment
# MAGIC If neither metric nor iteration are specified in the register_model call, the iteration with the best primary metric is registered.

# COMMAND ----------

description = 'Malaria DREAM Challenge 2019 (Subchallenge 1) AutoML Model'
tags = None
model = local_run.register_model(description = description, tags = tags)

print(local_run.model_id) # This will be written to the script file later

# COMMAND ----------

# MAGIC %md Create Scoring Script

# COMMAND ----------

# MAGIC %%writefile scoreclass.py
# MAGIC import pickle
# MAGIC import json
# MAGIC import numpy
# MAGIC import azureml.train.automl
# MAGIC from sklearn.externals import joblib
# MAGIC from azureml.core.model import Model
# MAGIC 
# MAGIC 
# MAGIC def init():
# MAGIC     global model
# MAGIC     model_path = Model.get_model_path(model_name = local_run.model_id) # this name is model.id of model that we want to deploy
# MAGIC     # deserialize the model file back into a sklearn model
# MAGIC     model = joblib.load(model_path)
# MAGIC 
# MAGIC def run(rawdata):
# MAGIC     try:
# MAGIC         data = json.loads(rawdata)['data']
# MAGIC         data = numpy.array(data)
# MAGIC         result = model.predict(data)
# MAGIC     except Exception as e:
# MAGIC         result = str(e)
# MAGIC         return json.dumps({"error": result})
# MAGIC     return json.dumps({"result":result.tolist()})

# COMMAND ----------

# MAGIC %sh ls

# COMMAND ----------

# MAGIC %md Create a YAML File for the Environment

# COMMAND ----------

experiment = Experiment(ws, experiment_name)
ml_run = AutoMLRun(experiment = experiment, run_id = local_run.id)

# COMMAND ----------

dependencies = ml_run.get_run_sdk_dependencies(iteration = 7)

# COMMAND ----------

#for p in ['azureml-train-automl', 'azureml-sdk', 'azureml-core']:
#    print('{}\t{}'.format(p, dependencies[p]))

# COMMAND ----------

from azureml.core.conda_dependencies import CondaDependencies

myenv = CondaDependencies.create(conda_packages=['numpy','scikit-learn'], pip_packages=['azureml-sdk[automl]'])

conda_env_file_name = 'myenvclass.yml'
myenv.save_to_file('.', conda_env_file_name)

# COMMAND ----------

# Substitute the actual version number in the environment file.
# This is not strictly needed in this notebook because the model should have been generated using the current SDK version.
# However, we include this in case this code is used on an experiment from a previous SDK version.

with open(conda_env_file_name, 'r') as cefr:
    content = cefr.read()

#with open(conda_env_file_name, 'w') as cefw:
#    cefw.write(content.replace(azureml.core.VERSION, dependencies['azureml-sdk']))

# Substitute the actual model id in the script file.

script_file_name = 'scoreclass.py'

with open(script_file_name, 'r') as cefr:
    content = cefr.read()

with open(script_file_name, 'w') as cefw:
    cefw.write(content.replace('AutoML9c57a1b08best', local_run.model_id))

# COMMAND ----------

# MAGIC %md Create a Container Image

# COMMAND ----------

from azureml.core.image import Image, ContainerImage

image_config = ContainerImage.image_configuration(runtime= "python",
                                 execution_script = script_file_name,
                                 conda_file = conda_env_file_name,
                                 tags = {'area': "AppTriage", 'type': "automl_classification"},
                                 description = "Image for AppTriage automl classification")

image = Image.create(name = "apptriageautomlimage",
                     # this is the model object 
                     models = [model],
                     image_config = image_config, 
                     workspace = ws)

image.wait_for_creation(show_output = True)

if image.creation_state == 'Failed':
    print("Image build log at: " + image.image_build_log_uri)

# COMMAND ----------

# MAGIC %md Deploy the Image as a Web Service on Azure Container Instance

# COMMAND ----------

from azureml.core.webservice import AciWebservice

aciconfig = AciWebservice.deploy_configuration(cpu_cores = 1, 
                                               memory_gb = 1, 
                                               tags = {'area': "AppTriage", 'type': "automl_classification"}, 
                                               description = 'Service for AppTriage Automl Classification')

# COMMAND ----------

from azureml.core.webservice import Webservice

aci_service_name = 'apptriage-automl'
print(aci_service_name)
aci_service = Webservice.deploy_from_image(deployment_config = aciconfig,
                                           image = image,
                                           name = aci_service_name,
                                           workspace = ws)
aci_service.wait_for_deployment(True)
print(aci_service.state)

# COMMAND ----------

# MAGIC %md Delete a Web Service

# COMMAND ----------

#aci_service.delete()

# COMMAND ----------

# MAGIC %md Get Logs from a Deployed Web Service

# COMMAND ----------

#aci_service.get_logs()

# COMMAND ----------

# DBTITLE 1,Test the Webservice
# for index in np.random.choice(len(y_test), 3, replace = False):
#     print(index)
#     test_sample = json.dumps({'data':X_test[index:index + 1].tolist()})
#     print(test_sample)
#     predicted = aci_service.run(input_data = test_sample)
#     label = y_test[index]
#     predictedDict = json.loads(predicted)
#     title = "Label value = %d  Predicted value = %s " % ( label,predictedDict['result'][0])

# COMMAND ----------

import requests
#data = u'{"var_0":-0.0770664853131316,"var_1":0,"var_2":0,"var_3":0,"var_4":"c_0","var_5":"c_1","var_6":0,"var_7":0.110331003825081,"var_8":-0.495910730559406,"var_9":0,"var_10":-0.485996022409892,"var_11":1.27163358277212,"var_12":0,"var_13":0.0155223063168767,"var_14":0,"var_15":0,"var_16":0,"var_17":0,"var_18":-0.452684225358065,"var_19":-0.247513322679647,"var_20":0,"var_21":-0.67669193529911,"var_22":0,"var_23":0,"var_24":0.822025562174633,"var_25":1,"var_26":"c_2","var_27":0,"var_28":0,"var_29":0,"var_30":0,"var_31":0.51320877456149,"var_32":"c_1","var_33":0,"var_34":"c_5","var_35":-1.06558939161236,"var_36":0,"var_37":"c_0","var_38":-0.210201975015424,"var_39":1,"var_40":0,"var_41":0}'

data = u'{"data": [[1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, -0.2809537293837132, 1.2530906708035863, 0.11033100382508193, 0.0, 0.0, 1.0, 1.0, -0.6651774606393488, 3.235483428798657, -0.4859960224098927, -0.24751332267964707, 0.6975950623496785, -0.21020197501542454, -0.06418812733473409, 0.038940695952612035, -0.4526842253580652, 0.015522306316876792]]}'

r = requests.post(data=data, url="http://52.191.234.133/score", headers = { 'Content-Type':'application/json' })
print("Response {}".format(r.text))

# COMMAND ----------

