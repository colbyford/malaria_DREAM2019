# Ensemble Machine Learning Modeling for the Prediction of Artemisinin Resistance in Malaria
<p align = "right">By: Colby T. Ford, Ph.D., Tamar Carter, Ph.D., and Daniel Janies, Ph.D.</p>

<details>
<summary><strong><em>Table of Contents</em></strong></summary>
* [Malaria DREAM Challenge 2019](#malaria-dream-challenge-2019)
  - [About the Challenge](#about-the-challenge)
  - [Subchallenge 1](#subchallenge-1)
  - [Subchallenge 2](#subchallenge-2)
</details>

# Malaria DREAM Challenge 2019
<h1 align = "center">🦟 👨‍🔬 🧬</h1>

<p align = "right">Submission by: Colby T. Ford, Ph.D.</p>

## About the Challenge

[Challenge Link](https://www.synapse.org/#!Synapse:syn16924919/wiki/)

The Malaria DREAM Challenge is open to anyone interested in contributing to the development of computational models that address important problems in advancing the fight against malaria. The overall goal of the first Malaria DREAM Challenge is to predict Artemisinin (Art) drug resistance level of a test set of malaria parasites using their in vitro transcription data and a training set consisting of published in vivo and unpublished in vitro transcriptomes. The in vivo dataset consists of ~1000 transcription samples from various geographic locations covering a wide range of life cycles and resistance levels, with other accompanying data such as patient age, geographic location, Art combination therapy used, etc [Mok et al., (2015) Science]. The in vitro transcription dataset consists of 55 isolates, with transcription collected at two timepoints (6 and 24 hours post-invasion), in the absence or presence of an Art perturbation, for two biological replicates using a custom microarray at the Ferdig lab. Using these transcription datasets, participants will be asked to predict three different resistance states of a subset of the 55 in vitro isolate samples; 

## Subchallenge 1

__Topic:__ Predict the Artemisinin (Art) IC50 (drug concentration at which 50% of parasites die) of malaria isolates using in vitro transcriptomics data.

__Data:__ Transcription profiles of all 55 malaria parasite isolates along with the IC50 (drug concentration at which 50% of parasites die) values.
 - Training set: 30 malaria parasite isolates
 - Test Set: 25 malaria parasite isolates.

The transcription data will consist of roughly 5000 genes from the malaria parasite, _Plasmodium falciparum_. For each malaria parasite isolate, transcription data was collected at two time points (6 hours post invasion (hpi) and 24hpi), perturbed and un-perturbed with dihydroartemisinin (the metabolically active form of artemisinin), with a biological replicate each, for a grand total of at least eight data points per parasite isolate.

| (Adapted with permission from Turnbull et al., (2017) PLoS One) | Training Set |        Test Set       |
|:---------------------------------------------------------------:|:------------:|:---------------------:|
|  Array                                                          | Bozdech      | Agilent HD Exon Array |
| Platform                                                        | Printed      | Agilent               |
| Plexes                                                          | 1            | 8                     |
| Unique Probes                                                   | 10159        | 62976                 |
| Range of Probes per Exon                                        | N/A          | 1-52                  |
| Average Probes per Gene                                         | 2            | 12                    |
| Genes Represented                                               | 5363         | 5440                  |
| Transcript Isoform Profiling                                    | No           | Yes                   |
| ncRNAs                                                          | No           | Yes                   |
| Channel Detection Method                                        | Two Color    | Single Color          |
| Scanner                                                         | PowerScanner | Agilent               |
| Data Extraction                                                 | GenePix Pro  | Agilent               |


### July 31, 2019 Submission:

**Data Preprocessing:** Casted data to roll up to the Isolate grain. Instead of having multiple rows for an isolate, each with various Timepoints, Treatments, and BioReps, this was pivoted such that each row represents a single isolate. So, each numerical column turns into 8 columns, one version for each slice of 2 Timepoints, 2 Treatments, and 2 BioReps. This resulted in a dataset with 44,346 columns.

**Model:** Voting Ensemble model (using soft voting) of 498 previous models (including models trained using Extreme Random Trees, Random Forest, Decision Tree, Elastic Net, and Lasso Least Angle Regression), each having been trained on pre-processed data using various steps (including imputation, normalization, and scaling).

**Metrics:**

- Normalized Mean Absolute Error: 0.1097
- Normalized Root Mean Squared Error: 0.1228

**Rank:** ? out of ??

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

__Topic:__ Utilizing a previously published _in vivo_ transcription data set from [Mok et al,. Science 2015](https://www.ncbi.nlm.nih.gov/pubmed/25502316), predict the resistance status of malaria isolates utilizing _in vitro_ transcription data.

__Data:__ _in vivo_ transcription data set ([Mok et al,. Science 2015](https://www.ncbi.nlm.nih.gov/pubmed/25502316)) to predict the parasite clearance rate of malaria parasite isolates based on in vitro transcriptional profiles.

|                               |                           Training Set                           |                                          Test Set                                          |
|:-----------------------------:|:----------------------------------------------------------------:|:------------------------------------------------------------------------------------------:|
| Dataset                       | Published from                                                   | Unpublished                                                                                |
| Number of Isolates            | 1043                                                             | 32                                                                                         |
| Isolate Collection Site       | Southeast Asia                                                   | Thai-Myanmar Border                                                                        |
| Isolate Collection Years      | 2012-2014                                                        | 2007-2012                                                                                  |
| Sample Type                   | _in vivo_                                                        | _in vitro_                                                                                 |
| Synchronized?                 | Not Synchronized                                                 | Synchronized                                                                               |
| Number of Samples per Isolate | 1                                                                | 8                                                                                          |
| Additional Attributes         |  ~18 Hours Post Invasion (hpi), Non-perturbed, No replicates | Two time points (6 hpi and 24 hpi), Perturbed and Non-perturbed, Biological Replicate |

### July 31, 2019 Submission:

**Data Preprocessing:** None, except removal of certain attributes. (`Country` - test data only contains "Thailand_Myanmar_Border", which isn't in the training data, `Kmeans_Grp` - not included in the testing data, and `Asexual_stage__hpi_` - testing data has different stages.)

**Model:** Voting Ensemble model (using soft voting) of 98 previous models (including models trained using Logistic Regression, SVM, Gradient Boosting, SGD, Naïve Bayes, ~~KNN~~, Random Forests, and Extreme Random Trees), each having been trained on pre-processed data using various steps (including imputation, normalization, and scaling).

**Metrics:**

- Weighted AUC: 0.8705
- Weighted F1 Score: 0.8019
- Weighted Accuracy: 0.8585

**Rank:** ?? out of ??

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
