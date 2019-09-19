# Malaria DREAM Challenge 2019
## Submission Rankings

<p align = "right">Submission by: Colby T. Ford, Ph.D.</p>

[Challenge Link](https://www.synapse.org/#!Synapse:syn16924919/wiki/)


## Subchallenge 1

### July 31, 2019 Submission:

**Data Preprocessing:** Casted data to roll up to the Isolate grain. Instead of having multiple rows for an isolate, each with various Timepoints, Treatments, and BioReps, this was pivoted such that each row represents a single isolate. So, each numerical column turns into 8 columns, one version for each slice of 2 Timepoints, 2 Treatments, and 2 BioReps. This resulted in a dataset with 44,346 columns.

**Model:** Voting Ensemble model (using soft voting) of 498 previous models (including models trained using Extreme Random Trees, Random Forest, Decision Tree, Elastic Net, and Lasso Least Angle Regression), each having been trained on pre-processed data using various steps (including imputation, normalization, and scaling).

**Metrics:**

- Normalized Mean Absolute Error: 0.1097
- Normalized Root Mean Squared Error: 0.1228

**Rank:** 7 out of 18

### July 18, 2019 Submission:

**Data Preprocessing:** Casted data to roll up to the Isolate grain. Instead of having multiple rows for an isolate, each with various Timepoints, Treatments, and BioReps, this was pivoted such that each row represents a single isolate. So, each numerical column turns into 8 columns, one version for each slice of 2 Timepoints, 2 Treatments, and 2 BioReps. This resulted in a dataset with 44,346 columns.

**Model:** Voting Ensemble model (using soft voting) of 97 previous models (including models trained using Extreme Random Trees, Random Forest, Decision Tree, Elastic Net, and Lasso Least Angle Regression), each having been trained on pre-processed data using various steps (including imputation, normalization, and scaling).

**Metrics:**

- Pearson Correlation: 0.65
- Normalized Mean Absolute Error: 0.1444
- Normalized Root Mean Squared Error: 0.1943

**Rank:** 4 out of 19

### July 11, 2019 Submission:

**Data Preprocessing:** Casted data to roll up to the Isolate grain. Instead of having multiple rows for an isolate, each with various Timepoints, Treatments, and BioReps, this was pivoted such that each row represents a single isolate. So, each numerical column turns into 8 columns, one version for each slice of 2 Timepoints, 2 Treatments, and 2 BioReps. This resulted in a dataset with 44,346 columns.

**Model:** Voting Ensemble model (using soft voting) of 23 previous models (including models trained using Extreme Random Trees, Random Forest, Decision Tree, Elastic Net, and Lasso Least Angle Regression), each having been trained on pre-processed data using various steps (including imputation, normalization, and scaling).

**Metrics:**

- Normalized Mean Absolute Error: 0.1479
- Normalized Root Mean Squared Error: 0.1871

**Rank:** 2 out of 17

## Subchallenge 2

### July 31, 2019 Submission:

**Data Preprocessing:** None, except removal of certain attributes. (`Country` - test data only contains "Thailand_Myanmar_Border", which isn't in the training data, `Kmeans_Grp` - not included in the testing data, and `Asexual_stage__hpi_` - testing data has different stages.)

**Model:** Voting Ensemble model (using soft voting) of 98 previous models (including models trained using Logistic Regression, SVM, Gradient Boosting, SGD, Naïve Bayes, ~~KNN~~, Random Forests, and Extreme Random Trees), each having been trained on pre-processed data using various steps (including imputation, normalization, and scaling).

**Metrics:**

- Weighted AUC: 0.8705
- Weighted F1 Score: 0.8019
- Weighted Accuracy: 0.8585

**Rank:** 3 out of 17

### July 18, 2019 Submission:

**Data Preprocessing:** None, except removal of certain attributes. (`Country` - test data only contains "Thailand_Myanmar_Border", which isn't in the training data, `Kmeans_Grp` - not included in the testing data, and `Asexual_stage__hpi_` - testing data has different stages.)

**Model:** Voting Ensemble model (using soft voting) of 97 previous models (including models trained using Logistic Regression, SVM, Gradient Boosting, SGD, Naïve Bayes, KNN, Random Forests, and Extreme Random Trees), each having been trained on pre-processed data using various steps (including imputation, normalization, and scaling).

**Metrics:**

- Weighted AUC: 0.8671
- Weighted F1 Score: 0.8024
- Weighted Accuracy: 0.8543

**Rank:** 5 out of 15

### July 11, 2019 Submission:

**Data Preprocessing:** None, except removal of certain attributes. (`Country` - test data only contains "Thailand_Myanmar_Border", which isn't in the training data, `Kmeans_Grp` - not included in the testing data, and `Asexual_stage__hpi_` - testing data has different stages.)

**Model:** Voting Ensemble model (using soft voting) of 48 previous models (including models trained using Logistic Regression, SVM, Gradient Boosting, SGD, Naïve Bayes, KNN, Random Forests, and Extreme Random Trees), each having been trained on pre-processed data using various steps (including imputation, normalization, and scaling).

**Metrics:**

- Weighted AUC: 0.8693
- Weighted F1 Score: 0.7930
- Weighted Accuracy: 0.8631

**Rank:** 2 out of 17
