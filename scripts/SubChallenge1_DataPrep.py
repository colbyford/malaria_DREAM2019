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

# column_list = train.columns
# prefix = "my_prefix"
# # new_column_list = [prefix + s for s in column_list]
# new_column_list = [prefix + s if s != "Isolate" else s for s in column_list]
 
# column_mapping = [[o, n] for o, n in zip(column_list, new_column_list)]

# print(column_mapping)

# # train2 = train.select(list(map(lambda old, new: col(old).alias(new),*zip(*column_mapping))))
# # display(train2)

# COMMAND ----------

# DBTITLE 1,Reshape Data
from pyspark.sql.functions import first, col

## Separate Dependent Variable
y = train.select(col("Isolate"),
                 col("DHA_IC50")) \
         .distinct()

############################################################################
print("1. Create Slice [Timepoint: 24HR, Treatment: DHA, BioRep: BRep1]")
hr24_trDHA_br1 = train.drop("Sample_Name","DHA_IC50") \
                      .filter((col("Timepoint") == "24HR") &
                              (col("Treatment") == "DHA") &
                              (col("BioRep") == "BRep1"))
## Rename Columns
column_list = hr24_trDHA_br1.columns
prefix = "hr24_trDHA_br1_"
new_column_list = [prefix + s if s != "Isolate" else s for s in column_list]

column_mapping = [[o, n] for o, n in zip(column_list, new_column_list)]

hr24_trDHA_br1 = hr24_trDHA_br1.select(list(map(lambda old, new: col(old).alias(new),*zip(*column_mapping))))
############################################################################

############################################################################
print("2. Create Slice [Timepoint: 24HR, Treatment: DHA, BioRep: BRep2]")
hr24_trDHA_br2 = train.drop("Sample_Name","DHA_IC50") \
                      .filter((col("Timepoint") == "24HR") &
                              (col("Treatment") == "DHA") &
                              (col("BioRep") == "BRep2"))
## Rename Columns
column_list = hr24_trDHA_br2.columns
prefix = "hr24_trDHA_br2_"
new_column_list = [prefix + s if s != "Isolate" else s for s in column_list]

column_mapping = [[o, n] for o, n in zip(column_list, new_column_list)]

hr24_trDHA_br2 = hr24_trDHA_br2.select(list(map(lambda old, new: col(old).alias(new),*zip(*column_mapping))))
############################################################################

############################################################################
print("3. Create Slice [Timepoint: 24HR, Treatment: UT, BioRep: BRep1]")
hr24_trUT_br1 = train.drop("Sample_Name","DHA_IC50") \
                      .filter((col("Timepoint") == "24HR") &
                              (col("Treatment") == "UT") &
                              (col("BioRep") == "BRep1"))
## Rename Columns
column_list = hr24_trUT_br1.columns
prefix = "hr24_trUT_br1_"
new_column_list = [prefix + s if s != "Isolate" else s for s in column_list]

column_mapping = [[o, n] for o, n in zip(column_list, new_column_list)]

hr24_trUT_br1 = hr24_trUT_br1.select(list(map(lambda old, new: col(old).alias(new),*zip(*column_mapping))))
############################################################################

############################################################################
print("4. Create Slice [Timepoint: 24HR, Treatment: UT, BioRep: BRep2]")
hr24_trUT_br2 = train.drop("Sample_Name","DHA_IC50") \
                      .filter((col("Timepoint") == "24HR") &
                              (col("Treatment") == "UT") &
                              (col("BioRep") == "BRep2"))
## Rename Columns
column_list = hr24_trUT_br2.columns
prefix = "hr24_trUT_br2_"
new_column_list = [prefix + s if s != "Isolate" else s for s in column_list]

column_mapping = [[o, n] for o, n in zip(column_list, new_column_list)]

hr24_trUT_br2 = hr24_trUT_br2.select(list(map(lambda old, new: col(old).alias(new),*zip(*column_mapping))))
############################################################################

############################################################################
print("5. Create Slice [Timepoint: 6HR, Treatment: DHA, BioRep: BRep1]")
hr6_trDHA_br1 = train.drop("Sample_Name","DHA_IC50") \
                      .filter((col("Timepoint") == "6HR") &
                              (col("Treatment") == "DHA") &
                              (col("BioRep") == "BRep1"))
## Rename Columns
column_list = hr6_trDHA_br1.columns
prefix = "hr6_trDHA_br1_"
new_column_list = [prefix + s if s != "Isolate" else s for s in column_list]

column_mapping = [[o, n] for o, n in zip(column_list, new_column_list)]

hr6_trDHA_br1 = hr6_trDHA_br1.select(list(map(lambda old, new: col(old).alias(new),*zip(*column_mapping))))
############################################################################

############################################################################
print("6. Create Slice [Timepoint: 6HR, Treatment: DHA, BioRep: BRep2]")
hr6_trDHA_br2 = train.drop("Sample_Name","DHA_IC50") \
                      .filter((col("Timepoint") == "6HR") &
                              (col("Treatment") == "DHA") &
                              (col("BioRep") == "BRep2"))
## Rename Columns
column_list = hr6_trDHA_br2.columns
prefix = "hr6_trDHA_br2_"
new_column_list = [prefix + s if s != "Isolate" else s for s in column_list]

column_mapping = [[o, n] for o, n in zip(column_list, new_column_list)]

hr6_trDHA_br2 = hr6_trDHA_br2.select(list(map(lambda old, new: col(old).alias(new),*zip(*column_mapping))))
############################################################################

############################################################################
print("7. Create Slice [Timepoint: 6HR, Treatment: UT, BioRep: BRep1]")
hr6_trUT_br1 = train.drop("Sample_Name","DHA_IC50") \
                      .filter((col("Timepoint") == "6HR") &
                              (col("Treatment") == "UT") &
                              (col("BioRep") == "BRep1"))
## Rename Columns
column_list = hr6_trUT_br1.columns
prefix = "hr6_trUT_br1_"
new_column_list = [prefix + s if s != "Isolate" else s for s in column_list]

column_mapping = [[o, n] for o, n in zip(column_list, new_column_list)]

hr6_trUT_br1 = hr6_trUT_br1.select(list(map(lambda old, new: col(old).alias(new),*zip(*column_mapping))))
############################################################################

############################################################################
print("8. Create Slice [Timepoint: 6HR, Treatment: UT, BioRep: BRep2]")
hr6_trUT_br2 = train.drop("Sample_Name","DHA_IC50") \
                      .filter((col("Timepoint") == "6HR") &
                              (col("Treatment") == "UT") &
                              (col("BioRep") == "BRep2"))
## Rename Columns
column_list = hr6_trUT_br2.columns
prefix = "hr6_trUT_br2_"
new_column_list = [prefix + s if s != "Isolate" else s for s in column_list]

column_mapping = [[o, n] for o, n in zip(column_list, new_column_list)]

hr6_trUT_br2 = hr6_trUT_br2.select(list(map(lambda old, new: col(old).alias(new),*zip(*column_mapping))))
############################################################################

## Join Slices Together
print("Joining all together...")
shaped_train = y.join(hr24_trDHA_br1, "Isolate", how='left') \
                .join(hr24_trDHA_br2, "Isolate", how='left') \
                .join(hr24_trUT_br1, "Isolate", how='left') \
                .join(hr24_trUT_br2, "Isolate", how='left') \
                .join(hr6_trDHA_br1, "Isolate", how='left') \
                .join(hr6_trDHA_br2, "Isolate", how='left') \
                .join(hr6_trUT_br1, "Isolate", how='left') \
                .join(hr6_trUT_br2, "Isolate", how='left') \

display(shaped_train)

# COMMAND ----------

# DBTITLE 1,Write to Blob
# shaped_train.coalesce(1).write.mode('overwrite').option("header", "true").csv("/mnt/malaria/SubCh1_TrainingData_Shaped.csv")

# COMMAND ----------

# DBTITLE 1,Define Columns
#Note: Columns with periods (.) in their names have been changed to underscore (_).

label = "DHA_IC50"

categoricalColumns = []

numericalColumns = shaped_train.drop("Isolate",
                                     "DHA_IC50",
                                     "hr24_trDHA_br1_Timepoint",
                                     "hr24_trDHA_br1_Treatment",
                                     "hr24_trDHA_br1_BioRep",
                                     "hr24_trDHA_br2_Timepoint",
                                     "hr24_trDHA_br2_Treatment",
                                     "hr24_trDHA_br2_BioRep",
                                     "hr24_trUT_br1_Timepoint",
                                     "hr24_trUT_br1_Treatment",
                                     "hr24_trUT_br1_BioRep",
                                     "hr24_trUT_br2_Timepoint",
                                     "hr24_trUT_br2_Treatment",
                                     "hr24_trUT_br2_BioRep",
                                     "hr6_trDHA_br1_Timepoint",
                                     "hr6_trDHA_br1_Treatment",
                                     "hr6_trDHA_br1_BioRep",
                                     "hr6_trDHA_br2_Timepoint",
                                     "hr6_trDHA_br2_Treatment",
                                     "hr6_trDHA_br2_BioRep",
                                     "hr6_trUT_br1_Timepoint",
                                     "hr6_trUT_br1_Treatment",
                                     "hr6_trUT_br1_BioRep",
                                     "hr6_trUT_br2_Timepoint",
                                     "hr6_trUT_br2_Treatment",
                                     "hr6_trUT_br2_BioRep").columns


# COMMAND ----------

# DBTITLE 1,Train Pipeline Model
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, OneHotEncoderEstimator, StringIndexer, VectorAssembler

categoricalColumnsclassVec = [c + "classVec" for c in categoricalColumns]

stages = []

for categoricalColumn in categoricalColumns:
  print(categoricalColumn)
  # Category Indexing with StringIndexer
  stringIndexer = StringIndexer(inputCol=categoricalColumn, outputCol = categoricalColumn+"Index").setHandleInvalid("skip")
  # Use OneHotEncoder to convert categorical variables into binary SparseVectors
  encoder = OneHotEncoder(inputCol=categoricalColumn+"Index", outputCol=categoricalColumn+"classVec")
  # Add stages.  These are not run here, but will run all at once later on.
  stages += [stringIndexer, encoder]

# Convert label into label indices using the StringIndexer
label_stringIndexer = StringIndexer(inputCol = label, outputCol = "label").setHandleInvalid("skip")
stages += [label_stringIndexer]

# Transform all features into a vector using VectorAssembler
assemblerInputs = categoricalColumnsclassVec + numericalColumns
assembler = VectorAssembler(inputCols = assemblerInputs, outputCol="features").setHandleInvalid("keep")
stages += [assembler]

prepPipeline = Pipeline().setStages(stages)
pipelineModel = prepPipeline.fit(shaped_train)
dataset = pipelineModel.transform(shaped_train)

# COMMAND ----------

# DBTITLE 1,Save Pipeline Model
dbutils.fs.rm("/mnt/malaria/sc1/pipeline", True)
pipelineModel.save("/mnt/malaria/sc1/pipeline")
display(dbutils.fs.ls("/mnt/malaria/sc1/pipeline"))

# COMMAND ----------

display(dataset.select("Isolate", "label","features"))

# COMMAND ----------

