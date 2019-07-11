# Databricks notebook source
# DBTITLE 1,Mount Blob Storage
# dbutils.fs.mount(
#   source = "wasbs://general@dsba6190storage.blob.core.windows.net/Malaria_DREAM19",
#   mount_point = "/mnt/malaria/",
#   extra_configs = {"fs.azure.account.key.dsba6190storage.blob.core.windows.net":"pliEE6me6O/ChaK5XSIaw4ijAH5AZGP7nng4AYc8hJY68icgnjrZ0D1P3jbQ1TLC8abFeTtG6nThN1VmkdihsA=="})

# COMMAND ----------

# MAGIC %md
# MAGIC # SubChallenge 1
# MAGIC 
# MAGIC ## Predict the Artemisinin (Art) IC50 (drug concentration at which 50% of parasites die) of malaria isolates using in vitro transcriptomics data.
# MAGIC 
# MAGIC __Scientific Rationale:__ Although IC50 measures are not representative of clinical resistance, insights into transcriptional changes that predict IC50 could clarify why we see a lack of IC50 shift in clinically resistant isolates, as well as perhaps a glimpse into the resistance mechanisms if clinical samples were to show an IC50 shift in the future.
# MAGIC 
# MAGIC ## a) Datasets Included:
# MAGIC 
# MAGIC The files listed in this folder include the following:
# MAGIC 
# MAGIC - SubCh1_TrainingData.csv
# MAGIC - SubCh1_TestData.csv
# MAGIC 
# MAGIC 
# MAGIC ## b) Layout/Description of Data
# MAGIC 
# MAGIC - SubCh1_TrainingData.csv and SubCh1_TestData.csv are the training and test data sets for the DREAM1 subchallenge 1 datasets, respectively. The training data set will be used to build a model to predict the IC50 of the test data sets. The training data set consists of 30 parasite isolates (or lines) whilst the test set consists of 25 parasite isolates. 
# MAGIC 
# MAGIC - Sample_Names = Unique identifier for each sample
# MAGIC 
# MAGIC - DHA_IC50 = Dihydroartemisinin IC50 (nM). IC50 of DHA, the metabolically active form of artemisinin. Located at the very last column of the dataset.
# MAGIC 
# MAGIC - Isolate = Parasite sample name. Each isolate is a parasite sample isolated from a single patient. 
# MAGIC 
# MAGIC - Timepoint = Estimated time after invasion which transcription sample was collected. Two options are available here 6hr or 24hr. Malaria has a 48hr life cycle in the blood stage and has very different development/transcription/biology at these two timepoints.
# MAGIC 
# MAGIC - Treatment = Whether the transcription sample was either perturbed with 5nM DHA or perturbed with DMSO (our control, listed as UT in dataset).
# MAGIC 
# MAGIC - BioRep = Biological Replicate. Indicates which biological replicate. Options available are Brep1 or Brep2.
# MAGIC 
# MAGIC - The transcription data was collected using a custom Agilent microarray. The transcription data set consists of MAL genes followed by PF3D7 genes with transcription values associated with each gene where available. The MAL genes are 92 non-coding RNAs while the PF3D7 genes are protein coding genes. An in-depth description about the microarray and its contents can be found here https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0187595 . We recommend using plasmodb.org/plasmo for descriptions about individual gene function. 

# COMMAND ----------

# DBTITLE 1,Read in Data
train = spark.read.format("csv") \
             .options(header = True, inferSchema = True) \
             .load("/mnt/malaria/SubCh1_TrainingData.csv")

display(train)

# COMMAND ----------

# DBTITLE 1,Define Columns
#Note: Columns with periods (.) in their names have been changed to underscore (_).

label = "DHA_IC50"

categoricalColumns = ["Timepoint", "Treatment", "BioRep"]

numericalColumns = train.drop("Sample_Name","Isolate","DHA_IC50","Timepoint","Treatment","BioRep").columns


# COMMAND ----------

# DBTITLE 1,Train Pipeline Model
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, OneHotEncoderEstimator, StringIndexer, VectorAssembler

categoricalColumnsclassVec = [c + "classVec" for c in categoricalColumns]

stages = []

for categoricalColumn in categoricalColumns:
  print(categoricalColumn)
  # Category Indexing with StringIndexer
  stringIndexer = StringIndexer(inputCol=categoricalColumn, outputCol = categoricalColumn+"Index").setHandleInvalid("keep")
  # Use OneHotEncoder to convert categorical variables into binary SparseVectors
  encoder = OneHotEncoder(inputCol=categoricalColumn+"Index", outputCol=categoricalColumn+"classVec")
  # Add stages.  These are not run here, but will run all at once later on.
  stages += [stringIndexer, encoder]

# Convert label into label indices using the StringIndexer
#label_stringIndexer = StringIndexer(inputCol = label, outputCol = "label").setHandleInvalid("skip")
#stages += [label_stringIndexer]

# Transform all features into a vector using VectorAssembler
assemblerInputs = categoricalColumnsclassVec + numericalColumns
assembler = VectorAssembler(inputCols = assemblerInputs, outputCol="features").setHandleInvalid("keep")
stages += [assembler]

prepPipeline = Pipeline().setStages(stages)
pipelineModel = prepPipeline.fit(train)
dataset = pipelineModel.transform(train)

# COMMAND ----------

# DBTITLE 1,Save Pipeline Model
dbutils.fs.rm("/mnt/malaria/sc1_noreshape/pipeline", True)
pipelineModel.save("/mnt/malaria/sc1_noreshape/pipeline")
display(dbutils.fs.ls("/mnt/malaria/sc1_noreshape/pipeline"))

# COMMAND ----------

# display(dataset.select("Isolate", "label","features"))

# COMMAND ----------

# MAGIC %md
# MAGIC -------------------------
# MAGIC ## Training Data

# COMMAND ----------

# DBTITLE 1,Data Prep - Load Data into Spark
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col

train = spark.read.format("csv") \
               .options(header = True, inferSchema = True) \
               .load("/mnt/malaria/SubCh1_TrainingData.csv")

pipeline = PipelineModel.load("/mnt/malaria/sc1_noreshape/pipeline/")

# COMMAND ----------

# DBTITLE 1,Transform Training Data Through Pipeline
data = pipeline.transform(train).select(col("DHA_IC50"), col("features")).withColumnRenamed("DHA_IC50","label")

display(data)

# COMMAND ----------

# DBTITLE 1,Data Prep - Convert Spark DataFrame to Numpy Array
import numpy as np

## Whole Training Data
pdtrain = data.toPandas()
trainseries = pdtrain['features'].apply(lambda x : np.array(x.toArray())).as_matrix().reshape(-1,1)
X_train = np.apply_along_axis(lambda x : x[0], 1, trainseries)
y_train = pdtrain['label'].values.reshape(-1,1).ravel()

print(y_train)

# COMMAND ----------

import pickle
pickle.dump(X_train, open( "sc1_noreshape_X_train.pkl", "wb" ) )
dbutils.fs.cp("file:/databricks/driver/sc1_noreshape_X_train.pkl", "/mnt/malaria/sc1_noreshape/arraydata")

pickle.dump(y_train, open( "sc1_noreshape_y_train.pkl", "wb" ) )
dbutils.fs.cp("file:/databricks/driver/sc1_noreshape_y_train.pkl", "/mnt/malaria/sc1_noreshape/arraydata")


display(dbutils.fs.ls("/mnt/malaria/sc1_noreshape/arraydata"))

# COMMAND ----------

# MAGIC %md
# MAGIC -------------------------
# MAGIC ## Test Data

# COMMAND ----------

# DBTITLE 1,Data Prep - Load Data into Spark
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col

test = spark.read.format("csv") \
               .options(header = True, inferSchema = True) \
               .load("/mnt/malaria/SubCh1_TestData.csv")

pipeline = PipelineModel.load("/mnt/malaria/sc1_noreshape/pipeline/")

# COMMAND ----------

# DBTITLE 1,Transform Test Through Pipeline
data = pipeline.transform(test).select(col("DHA_IC50"), col("features")).withColumnRenamed("DHA_IC50","label")

display(data)

# COMMAND ----------

# DBTITLE 1,Data Prep - Convert Spark DataFrame to Numpy Array
import numpy as np

## Whole Test Data
pdtest = data.toPandas()
testseries = pdtest['features'].apply(lambda x : np.array(x.toArray())).as_matrix().reshape(-1,1)
X_test = np.apply_along_axis(lambda x : x[0], 1, testseries)
#y_test = pdtest['label'].values.reshape(-1,1).ravel()

print(X_test)

# COMMAND ----------

import pickle
pickle.dump(X_test, open( "sc1_noreshape_X_test.pkl", "wb" ) )
dbutils.fs.cp("file:/databricks/driver/sc1_noreshape_X_test.pkl", "/mnt/malaria/sc1_noreshape/arraydata")

display(dbutils.fs.ls("/mnt/malaria/sc1_noreshape/arraydata"))