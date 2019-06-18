########################################################
## Script for reshaping Malaria Dataset               ##
## For Malaria DREAM Challenge 2019 - Subchallenge 1  ##
## Written by: Colby T. Ford, Ph.D.                   ##
########################################################

## Load in packages
library(dplyr)
library(tidyr)
library(readr)

## Load in data
train <- read_csv("../data/SubCh1_TrainingData.csv")

## Get value column names by which to pivot
valcols <- colnames(train %>% select(-c(Sample_Name, Isolate, Timepoint, Treatment, BioRep, DHA_IC50)))

## Reshape data: Gets 1 row per isolate, extends each Timepoint, Treatment, and BioRep version as its own column
train.reshaped <- train %>% 
  select(-c(Sample_Name, DHA_IC50)) %>%
  gather(variable, value, valcols) %>% 
  unite(variable, Timepoint, Treatment, BioRep, variable) %>% 
  spread(variable, value)

## Add back in the Depedent Variable
train.reshaped$DHA_IC50 <- train %>% select(c(Isolate, DHA_IC50)) %>% unique()

## Write out dataset
write_csv(train.reshaped, "../data/SubCh1_TrainingData_Reshaped.csv")



############################
## Machine Learning Models

## Load in data
train <- read_csv("../data/SubCh1_TrainingData.csv") %>% na.omit()

## Generalized Linear Model
glm.model <- glm(DHA_IC50 ~ .,
                 train %>%
                   select(-Sample_Name,
                          -Isolate) %>% 
                   na.omit(),
                 family = "gaussian")
library(caret)
varImp(glm.model)

## Random Forest

library(randomForest)
rf.model <- randomForest(DHA_IC50 ~ .,
                         train %>%
                           select(-Sample_Name,
                                  -Isolate) %>%
                           na.omit())
importance(fit_rf)