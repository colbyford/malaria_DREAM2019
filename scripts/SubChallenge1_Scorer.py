# Databricks notebook source
# MAGIC %md
# MAGIC # Malaria DREAM Challenge 2019
# MAGIC ## Subchallenge 1 - Scorer
# MAGIC ------------------------------
# MAGIC 
# MAGIC ## Transform Data into Arrays

# COMMAND ----------

# DBTITLE 1,Data Prep - Load Data into Spark
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col

test = spark.read.format("csv") \
               .options(header = True, inferSchema = True) \
               .load("/mnt/malaria/SubCh1_TestData.csv")

pipeline = PipelineModel.load("/mnt/malaria/sc1/pipeline/")

# COMMAND ----------

# DBTITLE 1,Reshape Data
from pyspark.sql.functions import first, col

## Separate Dependent Variable
y = test.select(col("Isolate"),
                 col("DHA_IC50")) \
         .distinct()

############################################################################
print("1. Create Slice [Timepoint: 24HR, Treatment: DHA, BioRep: BRep1]")
hr24_trDHA_br1 = test.drop("Sample_Name","DHA_IC50") \
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
hr24_trDHA_br2 = test.drop("Sample_Name","DHA_IC50") \
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
hr24_trUT_br1 = test.drop("Sample_Name","DHA_IC50") \
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
hr24_trUT_br2 = test.drop("Sample_Name","DHA_IC50") \
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
hr6_trDHA_br1 = test.drop("Sample_Name","DHA_IC50") \
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
hr6_trDHA_br2 = test.drop("Sample_Name","DHA_IC50") \
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
hr6_trUT_br1 = test.drop("Sample_Name","DHA_IC50") \
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
hr6_trUT_br2 = test.drop("Sample_Name","DHA_IC50") \
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
data = y.join(hr24_trDHA_br1, "Isolate", how='left') \
        .join(hr24_trDHA_br2, "Isolate", how='left') \
        .join(hr24_trUT_br1, "Isolate", how='left') \
        .join(hr24_trUT_br2, "Isolate", how='left') \
        .join(hr6_trDHA_br1, "Isolate", how='left') \
        .join(hr6_trDHA_br2, "Isolate", how='left') \
        .join(hr6_trUT_br1, "Isolate", how='left') \
        .join(hr6_trUT_br2, "Isolate", how='left') \

#display(data)

# COMMAND ----------

# DBTITLE 1,Transform Data Through Pipeline
data = pipeline.transform(data).select(col("DHA_IC50"), col("features")).withColumnRenamed("DHA_IC50","label")
#train, test = data.randomSplit([0.75, 0.25], 1337)

# test = spark.read.format("csv") \
#                .options(header = True, inferSchema = True) \
#                .load("/mnt/malaria/SubCh1_TestData.csv")
# test = pipeline.transform(test).select(col("label"), col("features"))

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

# DBTITLE 1,Save Arrays to Blob as Pickles
import pickle
pickle.dump(X_test, open( "sc1_X_test.pkl", "wb" ) )
dbutils.fs.cp("file:/databricks/driver/sc1_X_test.pkl", "/mnt/malaria/sc1/arraydata")

display(dbutils.fs.ls("/mnt/malaria/sc1/arraydata"))

# COMMAND ----------

