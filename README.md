# Ensemble Machine Learning Modeling for the Prediction of Artemisinin Resistance in Malaria

<p align = "right">By: Colby T. Ford, Ph.D. and Daniel Janies, Ph.D.</p>
<p align = "right">F1000 Research 2020; doi: <a href = "https://doi.org/10.12688/f1000research.21539.1">https://doi.org/10.12688/f1000research.21539.1</a></p>

Data: [![DOI](https://zenodo.org/badge/189651502.svg)](https://zenodo.org/badge/latestdoi/189651502)

<details>
<summary><strong><em>Table of Contents</em></strong></summary>

* [Malaria DREAM Challenge 2019](#malaria-dream-challenge-2019)
  - [About the Challenge](#about-the-challenge)
  - [Overall Modeling Approach](#overall-modeling-approach)
  - [Subchallenge 1](#subchallenge-1)
  - [Subchallenge 2](#subchallenge-2)

</details>

## Abstract

Antiparasitic resistance in malaria is a growing concern affecting many areas of the eastern world. Since the emergence of artemisinin resistance in the late 2000s in Cambodia, research into the underlying mechanisms has been underway.

The 2019 Malaria Dream Challenge posited the task of developing computational models that address important problems in advancing the fight against malaria. The first goal was to accurately predict Artemisinin drug resistance levels of _Plasmodium falciparum_ isolates, quantified by the IC50 The second goal was to predict the parasite clearance rate of malaria parasite isolates based on _in vitro_ transcriptional profiles.

In this work, we develop novel methods for transforming isolate data and handling the tens of thousands of variables that result from these data transformation exercises. This is demonstrated by using massively parallel processing of the data vectorization for use in scalable machine learning. In addition, we show the utility of ensemble machine learning modeling for highly effective predictions of both goals of this challenge. This is demonstrated by the use of multiple machine learning algorithms combined with various scaling and normalization preprocessing steps. Then, using a voting ensemble, multiple models are combined to generate a final model prediction.

## How to Cite

```latex
@Article{10.12688/f1000research.21539.1,
AUTHOR = {Ford, C.T. and Janies, D.},
TITLE = {{Ensemble Machine Learning Modeling for the Prediction of Artemisinin Resistance in Malaria}},
JOURNAL = {F1000Research},
VOLUME = {9},
YEAR = {2020},
NUMBER = {62},
DOI = {10.12688/f1000research.21539.1}
}
```

# Malaria DREAM Challenge 2019

<h1 align = "center">🦟 👨‍🔬 🧬</h1>

<p align = "right">Submission by: Colby T. Ford, Ph.D.</p>

## About the Challenge

[Challenge Link](https://www.synapse.org/#!Synapse:syn16924919/wiki/)

The Malaria DREAM Challenge is open to anyone interested in contributing to the development of computational models that address important problems in advancing the fight against malaria. The overall goal of the first Malaria DREAM Challenge is to predict Artemisinin (Art) drug resistance level of a test set of malaria parasites using their in vitro transcription data and a training set consisting of published in vivo and unpublished in vitro transcriptomes. The in vivo dataset consists of ~1000 transcription samples from various geographic locations covering a wide range of life cycles and resistance levels, with other accompanying data such as patient age, geographic location, Art combination therapy used, etc. [Mok et al., (2015) Science]. The in vitro transcription dataset consists of 55 isolates, with transcription collected at two timepoints (6 and 24 hours post-invasion), in the absence or presence of an Art perturbation, for two biological replicates using a custom microarray at the Ferdig lab. Using these transcription datasets, participants will be asked to predict three different resistance states of a subset of the 55 in vitro isolate samples.

## Overall Modeling Approach

![See: https://github.com/colbyford/malaria_DREAM2019/raw/master/process.png](https://github.com/colbyford/malaria_DREAM2019/raw/master/process_sm.png)

**Data Preprocessing:** Using Apache Spark (Specifically v2.4.3 on Azure Databricks v5.4), the input training data from Synapse (`SubChX_TrainingData.csv`) was loaded from Azure Blob Storage into the Spark environment. Spark was used for data reshaping, including the casting/pivoting of the data in SubChallenge 1 and the omission of unneeded variables in both SubChallenges. In addition, Spark was utilized to vectorize the datasets for use in distributed machine learning model training. As these training datasets are highly dimensional, Spark can parallelize the vectorization exercises across nodes of a cluster, reducing the time it took to get the vectorized output data.

**Machine Learning Modeling:** Numerous machine learning models were trained for each SubChallenge, using various scaling techniques and algorithms (AutoML). Microsoft Azure Machine Learning Service was utilized as the tracking platform for retaining model performance metrics as the various models were generated. Ensembling of multiple models was also tested, usually resulting in a better overall model. The ensembling method used was the [Caruana ensemble selection algorithm](http://www.niculescu-mizil.org/papers/shotgun.icml04.revised.rev2.pdf).

## Subchallenge 1

__Topic:__ Predict the Artemisinin (Art) IC50 (drug concentration at which 50% of parasites die) of malaria isolates using in vitro transcriptomics data.

__Data:__ Transcription profiles of all 55 malaria parasite isolates along with the IC50 (drug concentration at which 50% of parasites die) values.

- Training set: 30 malaria parasite isolates
- Test Set: 25 malaria parasite isolates.

The transcription data consists of 5540 genes from the malaria parasite, _Plasmodium falciparum_. For each malaria parasite isolate, transcription data was collected at two time points (6 hours post invasion (hpi) and 24hpi), perturbed and un-perturbed with dihydroartemisinin (the metabolically active form of artemisinin), with a biological replicate each, for a grand total of at least eight data points per parasite isolate.

| (Adapted with permission from Turnbull et al., (2017) PLoS One) | Training Set | Test Set              |
|:---------------------------------------------------------------:|:------------:|:---------------------:|
| Array                                                           | Bozdech      | Agilent HD Exon Array |
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

## Subchallenge 1 Final Submission

**Data Preprocessing:** Casted data to roll up to the Isolate grain. Instead of having multiple rows for an isolate, each with various Timepoints, Treatments, and BioReps, this was pivoted such that each row represents a single isolate. So, each numerical column turns into 8 columns, one version for each slice of 2 Timepoints, 2 Treatments, and 2 BioReps. This resulted in a dataset with 44,346 columns.

- [Data Shaping Script (PySpark on Apache Spark)](https://github.com/colbyford/malaria_DREAM2019/blob/master/SubChallenge1/scripts/SubChallenge1_DataPrep.py)
- [Vectorized Training Data (Independent Variables)](https://github.com/colbyford/malaria_DREAM2019/blob/master/SubChallenge1/data/sc1_X_train.pkl)
- [Vectorized Training Data (Dependent Variable)](https://github.com/colbyford/malaria_DREAM2019/blob/master/SubChallenge1/data/sc1_y_train.pkl)

**Training Parameters:** The following are the training parameters for the Azure Machine Learning Service:

| Parameter                   | Value                              |
|:---------------------------:|:----------------------------------:|
| Task                        | Regression                         |
| Number of Iterations        | 500                                |
| Iteration Timeout (minutes) | 20                                 |
| Max Cores per Iteration     | 7                                  |
| Primary Metric              | Normalized Root Mean Squared Error |
| Preprocess Data?            | True                               |
| k-Fold Cross-Validations    | 20 folds                           |

- [Submission Script for SubChallenge 1 Data to AMLS](https://github.com/colbyford/malaria_DREAM2019/blob/master/SubChallenge1/scripts/Submit_SubCh1_to_AMLS.py)

**Model:** Voting Ensemble model (using soft voting) of 498 previous models (including models trained using Extreme Random Trees, Random Forest, Decision Tree, Elastic Net, and Lasso Least Angle Regression), each having been trained on pre-processed data using various steps (including imputation, normalization, and scaling).

- [Ensemble Model Pickle File](https://github.com/colbyford/malaria_DREAM2019/blob/master/SubChallenge1/model/amls_model_7-31-19/sc1_model.pkl)

### SubChallenge 1 Model Information:

**Model Date:** 7/31/2019

**Platform:** Azure Machine Learning Service

**Best Model:** `VotingEnsemble`

**Model Description:** Using previous iterations of the machine learning model training, a `SoftVoting Ensemble` model was created. This uses the average of the class probabilities of previous iterations.

| Metric                                 | Value   |
|:--------------------------------------:|:-------:|
| normalized_root_mean_squared_error     | 0.1228  |
| spearman_correlation                   | -0.2    |
| root_mean_squared_log_error            | 0.1336  |
| normalized_mean_absolute_error         | 0.1097  |
| mean_absolute_percentage_error         | 24.27   |
| r2_score                               | -0.1461 |
| normalized_median_absolute_error       | 0.1097  |
| root_mean_squared_error                | 0.3398  |
| explained_variance                     | -1.755  |
| normalized_root_mean_squared_log_error | 0.1379  |
| median_absolute_error                  | 0.3035  |
| mean_absolute_error                    | 0.3035  |

## All Previous Iterations:

| ITERATION | RUN_PREPROCESSOR      | RUN_ALGORITHM      | NORMALIZED_ROOT_MEAN_SQUARED_ERROR |
|:---------:|:---------------------:|:------------------:|:----------------------------------:|
| 498       |                       | VotingEnsemble     | 0.12283293                         |
| 370       | SparseNormalizer      | RandomForest       | 0.132003138                        |
| 432       | StandardScalerWrapper | LightGBM           | 0.133180215                        |
| 240       | SparseNormalizer      | RandomForest       | 0.133779391                        |
| 430       | StandardScalerWrapper | RandomForest       | 0.137084337                        |
| 65        | SparseNormalizer      | RandomForest       | 0.13884791                         |
| 56        | SparseNormalizer      | RandomForest       | 0.14417843                         |
| 68        | MaxAbsScaler          | ExtremeRandomTrees | 0.151925822                        |
| 470       | StandardScalerWrapper | RandomForest       | 0.152262231                        |
| 181       | MinMaxScaler          | LightGBM           | 0.15279075                         |
| 441       | MaxAbsScaler          | LightGBM           | 0.154242112                        |
| 146       | StandardScalerWrapper | LightGBM           | 0.154765718                        |
| 269       | MaxAbsScaler          | LightGBM           | 0.155926785                        |
| 148       | RobustScaler          | LightGBM           | 0.156469078                        |
| 195       | SparseNormalizer      | RandomForest       | 0.156832998                        |
| 177       | SparseNormalizer      | LightGBM           | 0.156868761                        |
| 483       | MaxAbsScaler          | LightGBM           | 0.157322286                        |
| 97        | SparseNormalizer      | RandomForest       | 0.157341412                        |
| 221       | RobustScaler          | RandomForest       | 0.157344169                        |
| 342       | RobustScaler          | LightGBM           | 0.157355502                        |
| 401       | MaxAbsScaler          | LightGBM           | 0.157370636                        |
| 225       | MaxAbsScaler          | RandomForest       | 0.157746614                        |
| 169       | MinMaxScaler          | LightGBM           | 0.158444403                        |
| 55        | MaxAbsScaler          | LightGBM           | 0.15888225                         |
| 63        | SparseNormalizer      | LightGBM           | 0.159163354                        |
| 253       | RobustScaler          | LightGBM           | 0.159226125                        |
| 38        | StandardScalerWrapper | LightGBM           | 0.15939807                         |
| 57        | RobustScaler          | LightGBM           | 0.159511359                        |
| 207       | StandardScalerWrapper | LightGBM           | 0.15972357                         |
| 217       | MaxAbsScaler          | GradientBoosting   | 0.159911304                        |
| 386       | SparseNormalizer      | RandomForest       | 0.160045568                        |
| 313       | StandardScalerWrapper | RandomForest       | 0.160488808                        |
| 270       | RobustScaler          | RandomForest       | 0.160635674                        |
| 385       | TruncatedSVDWrapper   | RandomForest       | 0.160640259                        |
| 369       | MinMaxScaler          | GradientBoosting   | 0.160913975                        |
| 48        | SparseNormalizer      | RandomForest       | 0.160951856                        |
| 0         | RobustScaler          | ElasticNet         | 0.161583928                        |
| 35        | StandardScalerWrapper | ElasticNet         | 0.161583928                        |
| 499       |                       | StackEnsemble      | 0.161583928                        |
| 16        | StandardScalerWrapper | ElasticNet         | 0.161729381                        |
| 309       | SparseNormalizer      | RandomForest       | 0.161854681                        |
| 161       | StandardScalerWrapper | LightGBM           | 0.161871201                        |
| 407       | StandardScalerWrapper | LightGBM           | 0.162073958                        |
| 347       | MaxAbsScaler          | LightGBM           | 0.162089424                        |
| 230       | SparseNormalizer      | RandomForest       | 0.162145042                        |
| 262       | StandardScalerWrapper | LightGBM           | 0.162193706                        |
| 21        | StandardScalerWrapper | ElasticNet         | 0.162318961                        |
| 196       | StandardScalerWrapper | LightGBM           | 0.162434403                        |
| 30        | MaxAbsScaler          | ElasticNet         | 0.162438888                        |
| 408       | RobustScaler          | LightGBM           | 0.16248918                         |
| 64        | StandardScalerWrapper | LightGBM           | 0.162541441                        |
| 326       | RobustScaler          | RandomForest       | 0.162724319                        |
| 300       | PCA                   | RandomForest       | 0.162841003                        |
| 185       | MinMaxScaler          | RandomForest       | 0.162942257                        |
| 294       | RobustScaler          | LightGBM           | 0.162987298                        |
| 53        | MaxAbsScaler          | LightGBM           | 0.163408663                        |
| 379       | RobustScaler          | ExtremeRandomTrees | 0.163544922                        |
| 265       | SparseNormalizer      | RandomForest       | 0.163570788                        |
| 352       | RobustScaler          | RandomForest       | 0.163583341                        |
| 200       | TruncatedSVDWrapper   | RandomForest       | 0.16394081                         |
| 344       | MaxAbsScaler          | LightGBM           | 0.164164669                        |
| 246       | RobustScaler          | RandomForest       | 0.164214501                        |
| 45        | MaxAbsScaler          | LightGBM           | 0.164775009                        |
| 362       | MinMaxScaler          | LightGBM           | 0.164813327                        |
| 317       | StandardScalerWrapper | LightGBM           | 0.164832763                        |
| 128       | RobustScaler          | RandomForest       | 0.16489587                         |
| 275       | TruncatedSVDWrapper   | RandomForest       | 0.164973758                        |
| 131       | MaxAbsScaler          | LightGBM           | 0.165040416                        |
| 321       | StandardScalerWrapper | LightGBM           | 0.165341668                        |
| 223       | MinMaxScaler          | LightGBM           | 0.165372259                        |
| 365       | SparseNormalizer      | RandomForest       | 0.16541378                         |
| 147       | SparseNormalizer      | RandomForest       | 0.165442899                        |
| 212       | RobustScaler          | RandomForest       | 0.165761688                        |
| 40        | MinMaxScaler          | LightGBM           | 0.165783768                        |
| 256       | MinMaxScaler          | LightGBM           | 0.165842417                        |
| 123       | MinMaxScaler          | LightGBM           | 0.165957073                        |
| 494       | StandardScalerWrapper | LightGBM           | 0.166025767                        |
| 219       | StandardScalerWrapper | ExtremeRandomTrees | 0.166296437                        |
| 289       | MinMaxScaler          | GradientBoosting   | 0.166318337                        |
| 120       | TruncatedSVDWrapper   | RandomForest       | 0.166421551                        |
| 87        | RobustScaler          | ExtremeRandomTrees | 0.166567654                        |
| 395       | MinMaxScaler          | RandomForest       | 0.166584276                        |
| 176       | MaxAbsScaler          | RandomForest       | 0.166584841                        |
| 47        | RobustScaler          | RandomForest       | 0.166667434                        |
| 288       | MaxAbsScaler          | GradientBoosting   | 0.166676087                        |
| 70        | SparseNormalizer      | RandomForest       | 0.166812176                        |
| 110       | TruncatedSVDWrapper   | RandomForest       | 0.166861419                        |
| 387       | StandardScalerWrapper | ExtremeRandomTrees | 0.166867162                        |
| 445       | PCA                   | RandomForest       | 0.16691194                         |
| 400       | StandardScalerWrapper | RandomForest       | 0.167127388                        |
| 471       | StandardScalerWrapper | GradientBoosting   | 0.167163878                        |
| 99        | StandardScalerWrapper | RandomForest       | 0.167178832                        |
| 271       | MaxAbsScaler          | GradientBoosting   | 0.167260508                        |
| 373       | RobustScaler          | LightGBM           | 0.167466557                        |
| 297       | StandardScalerWrapper | RandomForest       | 0.16748517                         |
| 364       | StandardScalerWrapper | GradientBoosting   | 0.167493127                        |
| 104       | MaxAbsScaler          | ExtremeRandomTrees | 0.167510183                        |
| 374       | MinMaxScaler          | LightGBM           | 0.167529195                        |
| 23        | StandardScalerWrapper | LightGBM           | 0.16757886                         |
| 136       | MaxAbsScaler          | LightGBM           | 0.167692986                        |
| 363       | StandardScalerWrapper | ExtremeRandomTrees | 0.16789355                         |
| 282       | RobustScaler          | ExtremeRandomTrees | 0.167903461                        |
| 396       | StandardScalerWrapper | LightGBM           | 0.167915082                        |
| 36        | MinMaxScaler          | LightGBM           | 0.167916353                        |
| 463       | MinMaxScaler          | LightGBM           | 0.168026061                        |
| 43        | MinMaxScaler          | GradientBoosting   | 0.168196455                        |
| 335       | SparseNormalizer      | RandomForest       | 0.168237765                        |
| 150       | RobustScaler          | RandomForest       | 0.168305929                        |
| 211       | MinMaxScaler          | LightGBM           | 0.168609919                        |
| 466       | MinMaxScaler          | RandomForest       | 0.168626485                        |
| 179       | MaxAbsScaler          | RandomForest       | 0.168710466                        |
| 208       | MaxAbsScaler          | LightGBM           | 0.168756276                        |
| 71        | MaxAbsScaler          | ExtremeRandomTrees | 0.168966696                        |
| 461       | MinMaxScaler          | RandomForest       | 0.169014953                        |
| 117       | MinMaxScaler          | LightGBM           | 0.169108334                        |
| 457       | StandardScalerWrapper | RandomForest       | 0.169227418                        |
| 152       | MinMaxScaler          | LightGBM           | 0.169306795                        |
| 89        | StandardScalerWrapper | LightGBM           | 0.169307528                        |
| 100       | StandardScalerWrapper | RandomForest       | 0.169323732                        |
| 118       | StandardScalerWrapper | GradientBoosting   | 0.169377771                        |
| 251       | MaxAbsScaler          | LightGBM           | 0.169457188                        |
| 190       | RobustScaler          | RandomForest       | 0.169457405                        |
| 29        | StandardScalerWrapper | LightGBM           | 0.169535205                        |
| 143       | MinMaxScaler          | LightGBM           | 0.169600153                        |
| 296       | MaxAbsScaler          | LightGBM           | 0.169615949                        |
| 114       | RobustScaler          | RandomForest       | 0.169755714                        |
| 427       | RobustScaler          | RandomForest       | 0.169768327                        |
| 138       | SparseNormalizer      | LightGBM           | 0.169846711                        |
| 130       | SparseNormalizer      | RandomForest       | 0.169887666                        |
| 292       | StandardScalerWrapper | RandomForest       | 0.169965588                        |
| 84        | SparseNormalizer      | ExtremeRandomTrees | 0.17002492                         |
| 380       | SparseNormalizer      | RandomForest       | 0.170083227                        |
| 140       | StandardScalerWrapper | RandomForest       | 0.1701403                          |
| 345       | SparseNormalizer      | RandomForest       | 0.170213164                        |
| 95        | MaxAbsScaler          | RandomForest       | 0.170347557                        |
| 156       | StandardScalerWrapper | RandomForest       | 0.170362745                        |
| 358       | MinMaxScaler          | GradientBoosting   | 0.170440465                        |
| 247       | StandardScalerWrapper | RandomForest       | 0.170542682                        |
| 60        | StandardScalerWrapper | RandomForest       | 0.170593492                        |
| 325       | MinMaxScaler          | RandomForest       | 0.170658126                        |
| 456       | StandardScalerWrapper | RandomForest       | 0.170722151                        |
| 468       | RobustScaler          | RandomForest       | 0.170821843                        |
| 482       | SparseNormalizer      | RandomForest       | 0.170895322                        |
| 239       | RobustScaler          | RandomForest       | 0.170899922                        |
| 266       | StandardScalerWrapper | RandomForest       | 0.170902101                        |
| 168       | StandardScalerWrapper | ExtremeRandomTrees | 0.17090634                         |
| 278       | RobustScaler          | LightGBM           | 0.171016306                        |
| 209       | RobustScaler          | GradientBoosting   | 0.171378809                        |
| 80        | TruncatedSVDWrapper   | RandomForest       | 0.171433269                        |
| 112       | MinMaxScaler          | RandomForest       | 0.171434094                        |
| 192       | StandardScalerWrapper | RandomForest       | 0.171466942                        |
| 137       | StandardScalerWrapper | ExtremeRandomTrees | 0.171533691                        |
| 419       | RobustScaler          | RandomForest       | 0.171616533                        |
| 405       | MinMaxScaler          | RandomForest       | 0.171624986                        |
| 375       | StandardScalerWrapper | RandomForest       | 0.171627256                        |
| 51        | MinMaxScaler          | RandomForest       | 0.171707376                        |
| 487       | StandardScalerWrapper | ExtremeRandomTrees | 0.171718804                        |
| 105       | StandardScalerWrapper | RandomForest       | 0.171741383                        |
| 467       | RobustScaler          | RandomForest       | 0.171842874                        |
| 475       | MinMaxScaler          | RandomForest       | 0.172041647                        |
| 346       | RobustScaler          | LightGBM           | 0.172118192                        |
| 473       | MaxAbsScaler          | LightGBM           | 0.172133297                        |
| 399       | MaxAbsScaler          | ExtremeRandomTrees | 0.172162184                        |
| 486       | MaxAbsScaler          | RandomForest       | 0.172176668                        |
| 224       | RobustScaler          | RandomForest       | 0.172213036                        |
| 429       | MinMaxScaler          | RandomForest       | 0.172292827                        |
| 403       | MaxAbsScaler          | GradientBoosting   | 0.172353308                        |
| 4         | StandardScalerWrapper | LightGBM           | 0.172373695                        |
| 493       | StandardScalerWrapper | RandomForest       | 0.172485121                        |
| 469       | RobustScaler          | RandomForest       | 0.172521024                        |
| 384       | MinMaxScaler          | ExtremeRandomTrees | 0.172579395                        |
| 431       | MinMaxScaler          | LightGBM           | 0.172606539                        |
| 423       | StandardScalerWrapper | ExtremeRandomTrees | 0.172666405                        |
| 350       | MinMaxScaler          | RandomForest       | 0.172764823                        |
| 295       | RobustScaler          | RandomForest       | 0.172820501                        |
| 303       | StandardScalerWrapper | GradientBoosting   | 0.172864886                        |
| 249       | StandardScalerWrapper | GradientBoosting   | 0.172940215                        |
| 222       | MinMaxScaler          | ExtremeRandomTrees | 0.172956265                        |
| 31        | StandardScalerWrapper | LightGBM           | 0.172969647                        |
| 201       | MinMaxScaler          | ExtremeRandomTrees | 0.173042778                        |
| 422       | StandardScalerWrapper | RandomForest       | 0.173049985                        |
| 215       | StandardScalerWrapper | RandomForest       | 0.173184592                        |
| 485       | RobustScaler          | RandomForest       | 0.173220933                        |
| 127       | RobustScaler          | RandomForest       | 0.173361529                        |
| 442       | MinMaxScaler          | RandomForest       | 0.173387536                        |
| 188       | SparseNormalizer      | RandomForest       | 0.17340466                         |
| 142       | MinMaxScaler          | RandomForest       | 0.173501317                        |
| 280       | StandardScalerWrapper | RandomForest       | 0.173552374                        |
| 398       | MaxAbsScaler          | RandomForest       | 0.173603635                        |
| 213       | RobustScaler          | RandomForest       | 0.173605882                        |
| 361       | RobustScaler          | ExtremeRandomTrees | 0.173652506                        |
| 359       | StandardScalerWrapper | ExtremeRandomTrees | 0.173721412                        |
| 119       | RobustScaler          | RandomForest       | 0.173736256                        |
| 404       | StandardScalerWrapper | RandomForest       | 0.173744704                        |
| 425       | StandardScalerWrapper | RandomForest       | 0.173748066                        |
| 333       | MaxAbsScaler          | RandomForest       | 0.173755466                        |
| 83        | MinMaxScaler          | RandomForest       | 0.173805355                        |
| 153       | StandardScalerWrapper | GradientBoosting   | 0.173814819                        |
| 151       | StandardScalerWrapper | ExtremeRandomTrees | 0.173908442                        |
| 438       | RobustScaler          | ExtremeRandomTrees | 0.174001407                        |
| 149       | StandardScalerWrapper | ExtremeRandomTrees | 0.174049679                        |
| 428       | StandardScalerWrapper | RandomForest       | 0.174083172                        |
| 98        | StandardScalerWrapper | LightGBM           | 0.174091103                        |
| 413       | MinMaxScaler          | ExtremeRandomTrees | 0.174127523                        |
| 157       | RobustScaler          | ExtremeRandomTrees | 0.174166226                        |
| 286       | MinMaxScaler          | ExtremeRandomTrees | 0.174187187                        |
| 202       | MinMaxScaler          | ExtremeRandomTrees | 0.174209704                        |
| 443       | RobustScaler          | GradientBoosting   | 0.174247327                        |
| 409       | MaxAbsScaler          | ExtremeRandomTrees | 0.174250055                        |
| 458       | RobustScaler          | RandomForest       | 0.174261671                        |
| 290       | MinMaxScaler          | RandomForest       | 0.17427183                         |
| 134       | MaxAbsScaler          | ExtremeRandomTrees | 0.174383201                        |
| 107       | RobustScaler          | RandomForest       | 0.174569934                        |
| 465       | MinMaxScaler          | RandomForest       | 0.174623865                        |
| 392       | MinMaxScaler          | RandomForest       | 0.174714883                        |
| 26        | MaxAbsScaler          | ExtremeRandomTrees | 0.174720517                        |
| 287       | StandardScalerWrapper | ExtremeRandomTrees | 0.174724236                        |
| 133       | StandardScalerWrapper | ExtremeRandomTrees | 0.174812605                        |
| 302       | StandardScalerWrapper | LightGBM           | 0.174829146                        |
| 327       | MinMaxScaler          | RandomForest       | 0.174842679                        |
| 78        | RobustScaler          | RandomForest       | 0.174847739                        |
| 393       | StandardScalerWrapper | RandomForest       | 0.174850228                        |
| 310       | SparseNormalizer      | RandomForest       | 0.174860231                        |
| 453       | RobustScaler          | RandomForest       | 0.174861994                        |
| 82        | MaxAbsScaler          | ExtremeRandomTrees | 0.174888596                        |
| 454       | MaxAbsScaler          | ExtremeRandomTrees | 0.174894184                        |
| 206       | StandardScalerWrapper | ExtremeRandomTrees | 0.174894844                        |
| 418       | StandardScalerWrapper | LightGBM           | 0.174895302                        |
| 433       | StandardScalerWrapper | KNN                | 0.174910718                        |
| 242       | StandardScalerWrapper | GradientBoosting   | 0.174931339                        |
| 279       | MaxAbsScaler          | ExtremeRandomTrees | 0.175052279                        |
| 307       | RobustScaler          | RandomForest       | 0.175053944                        |
| 180       | StandardScalerWrapper | RandomForest       | 0.175055737                        |
| 355       | StandardScalerWrapper | RandomForest       | 0.175109318                        |
| 90        | RobustScaler          | RandomForest       | 0.175134294                        |
| 437       | RobustScaler          | ExtremeRandomTrees | 0.175175786                        |
| 175       | StandardScalerWrapper | RandomForest       | 0.175233475                        |
| 46        | MaxAbsScaler          | RandomForest       | 0.175248442                        |
| 472       | StandardScalerWrapper | ExtremeRandomTrees | 0.175295191                        |
| 139       | RobustScaler          | RandomForest       | 0.175305444                        |
| 338       | MinMaxScaler          | RandomForest       | 0.175322152                        |
| 24        | MinMaxScaler          | RandomForest       | 0.175376339                        |
| 69        | MaxAbsScaler          | ExtremeRandomTrees | 0.175376749                        |
| 339       | StandardScalerWrapper | LightGBM           | 0.17538851                         |
| 394       | MinMaxScaler          | ExtremeRandomTrees | 0.175413013                        |
| 184       | StandardScalerWrapper | ExtremeRandomTrees | 0.175445679                        |
| 73        | MinMaxScaler          | LightGBM           | 0.175462963                        |
| 93        | RobustScaler          | RandomForest       | 0.175498166                        |
| 199       | MinMaxScaler          | RandomForest       | 0.175632165                        |
| 94        | MinMaxScaler          | RandomForest       | 0.175673316                        |
| 478       | StandardScalerWrapper | RandomForest       | 0.175874329                        |
| 144       | StandardScalerWrapper | RandomForest       | 0.17588299                         |
| 252       | MinMaxScaler          | ExtremeRandomTrees | 0.175915641                        |
| 440       | StandardScalerWrapper | RandomForest       | 0.175918695                        |
| 193       | RobustScaler          | RandomForest       | 0.175943794                        |
| 415       | RobustScaler          | RandomForest       | 0.176028671                        |
| 397       | StandardScalerWrapper | RandomForest       | 0.17605315                         |
| 389       | MaxAbsScaler          | ExtremeRandomTrees | 0.176064544                        |
| 34        | StandardScalerWrapper | RandomForest       | 0.176230035                        |
| 154       | MinMaxScaler          | RandomForest       | 0.176340521                        |
| 311       | RobustScaler          | GradientBoosting   | 0.176377347                        |
| 263       | MinMaxScaler          | GradientBoosting   | 0.176383021                        |
| 378       | MaxAbsScaler          | RandomForest       | 0.176418819                        |
| 1         | StandardScalerWrapper | ElasticNet         | 0.176553009                        |
| 426       | StandardScalerWrapper | ExtremeRandomTrees | 0.176630817                        |
| 81        | MaxAbsScaler          | RandomForest       | 0.17663726                         |
| 245       | RobustScaler          | RandomForest       | 0.176783719                        |
| 477       | MaxAbsScaler          | RandomForest       | 0.176875732                        |
| 159       | MaxAbsScaler          | ExtremeRandomTrees | 0.176885249                        |
| 377       | MinMaxScaler          | ExtremeRandomTrees | 0.176917413                        |
| 319       | RobustScaler          | ExtremeRandomTrees | 0.1769424                          |
| 76        | MaxAbsScaler          | ExtremeRandomTrees | 0.176972873                        |
| 238       | MinMaxScaler          | ExtremeRandomTrees | 0.176990051                        |
| 210       | SparseNormalizer      | RandomForest       | 0.177039704                        |
| 368       | RobustScaler          | RandomForest       | 0.177065928                        |
| 312       | StandardScalerWrapper | RandomForest       | 0.177124989                        |
| 234       | MinMaxScaler          | RandomForest       | 0.177160391                        |
| 61        | MinMaxScaler          | LightGBM           | 0.177307862                        |
| 141       | MaxAbsScaler          | ExtremeRandomTrees | 0.177413563                        |
| 268       | SparseNormalizer      | ExtremeRandomTrees | 0.177457569                        |
| 111       | RobustScaler          | RandomForest       | 0.177590302                        |
| 66        | MaxAbsScaler          | RandomForest       | 0.177739728                        |
| 50        | StandardScalerWrapper | LightGBM           | 0.177768112                        |
| 229       | RobustScaler          | RandomForest       | 0.177827641                        |
| 383       | StandardScalerWrapper | RandomForest       | 0.177851583                        |
| 291       | StandardScalerWrapper | ExtremeRandomTrees | 0.178202569                        |
| 484       | StandardScalerWrapper | KNN                | 0.178374988                        |
| 75        | MaxAbsScaler          | RandomForest       | 0.17845734                         |
| 372       | StandardScalerWrapper | LightGBM           | 0.178505147                        |
| 109       | MaxAbsScaler          | ExtremeRandomTrees | 0.178583584                        |
| 115       | SparseNormalizer      | RandomForest       | 0.178597171                        |
| 226       | RobustScaler          | ExtremeRandomTrees | 0.178599049                        |
| 233       | StandardScalerWrapper | LightGBM           | 0.178646236                        |
| 447       | StandardScalerWrapper | ExtremeRandomTrees | 0.178725534                        |
| 490       | MinMaxScaler          | RandomForest       | 0.178790368                        |
| 11        | MinMaxScaler          | ExtremeRandomTrees | 0.179004822                        |
| 462       | MaxAbsScaler          | ExtremeRandomTrees | 0.17900803                         |
| 323       | MaxAbsScaler          | RandomForest       | 0.179177306                        |
| 244       | StandardScalerWrapper | RandomForest       | 0.179221838                        |
| 496       | StandardScalerWrapper | ExtremeRandomTrees | 0.179244947                        |
| 298       | MinMaxScaler          | RandomForest       | 0.179279888                        |
| 330       | MinMaxScaler          | RandomForest       | 0.179307334                        |
| 59        | StandardScalerWrapper | GradientBoosting   | 0.179374745                        |
| 281       | MaxAbsScaler          | ExtremeRandomTrees | 0.179418045                        |
| 72        | StandardScalerWrapper | GradientBoosting   | 0.17943037                         |
| 124       | MinMaxScaler          | LightGBM           | 0.179483664                        |
| 341       | RobustScaler          | RandomForest       | 0.179593337                        |
| 32        | MinMaxScaler          | LightGBM           | 0.179735268                        |
| 164       | StandardScalerWrapper | RandomForest       | 0.17979062                         |
| 455       | StandardScalerWrapper | RandomForest       | 0.179800881                        |
| 88        | StandardScalerWrapper | LightGBM           | 0.179804518                        |
| 340       | StandardScalerWrapper | RandomForest       | 0.179908987                        |
| 236       | StandardScalerWrapper | LightGBM           | 0.179921396                        |
| 216       | StandardScalerWrapper | ExtremeRandomTrees | 0.179965443                        |
| 172       | StandardScalerWrapper | ExtremeRandomTrees | 0.179986647                        |
| 479       | StandardScalerWrapper | RandomForest       | 0.180050635                        |
| 336       | RobustScaler          | ExtremeRandomTrees | 0.180066692                        |
| 267       | RobustScaler          | ExtremeRandomTrees | 0.180067514                        |
| 241       | MaxAbsScaler          | ExtremeRandomTrees | 0.180116246                        |
| 450       | StandardScalerWrapper | RandomForest       | 0.180129457                        |
| 203       | StandardScalerWrapper | RandomForest       | 0.18013079                         |
| 42        | MaxAbsScaler          | GradientBoosting   | 0.180228765                        |
| 235       | MinMaxScaler          | RandomForest       | 0.180230615                        |
| 406       | RobustScaler          | LightGBM           | 0.180329045                        |
| 382       | MaxAbsScaler          | ExtremeRandomTrees | 0.180384471                        |
| 332       | RobustScaler          | GradientBoosting   | 0.180475793                        |
| 417       | SparseNormalizer      | RandomForest       | 0.18052427                         |
| 315       | StandardScalerWrapper | RandomForest       | 0.180526435                        |
| 96        | MinMaxScaler          | LightGBM           | 0.180625406                        |
| 351       | MaxAbsScaler          | RandomForest       | 0.180626825                        |
| 284       | StandardScalerWrapper | RandomForest       | 0.180643822                        |
| 283       | MaxAbsScaler          | LightGBM           | 0.180706343                        |
| 162       | RobustScaler          | RandomForest       | 0.180707564                        |
| 160       | MaxAbsScaler          | RandomForest       | 0.180749311                        |
| 293       | RobustScaler          | RandomForest       | 0.180832944                        |
| 183       | RobustScaler          | ExtremeRandomTrees | 0.180847271                        |
| 320       | StandardScalerWrapper | RandomForest       | 0.180900539                        |
| 165       | RobustScaler          | RandomForest       | 0.181145369                        |
| 125       | StandardScalerWrapper | RandomForest       | 0.181178918                        |
| 348       | RobustScaler          | ExtremeRandomTrees | 0.181223413                        |
| 171       | StandardScalerWrapper | ExtremeRandomTrees | 0.181389063                        |
| 449       | MinMaxScaler          | ExtremeRandomTrees | 0.181401615                        |
| 54        | MaxAbsScaler          | RandomForest       | 0.181402212                        |
| 390       | StandardScalerWrapper | RandomForest       | 0.181472106                        |
| 158       | StandardScalerWrapper | ExtremeRandomTrees | 0.181475852                        |
| 318       | RobustScaler          | RandomForest       | 0.181548546                        |
| 255       | SparseNormalizer      | RandomForest       | 0.181647663                        |
| 306       | StandardScalerWrapper | ExtremeRandomTrees | 0.181727486                        |
| 33        | StandardScalerWrapper | RandomForest       | 0.181730245                        |
| 388       | MinMaxScaler          | ExtremeRandomTrees | 0.181737204                        |
| 424       | MaxAbsScaler          | ExtremeRandomTrees | 0.181787448                        |
| 260       | RobustScaler          | RandomForest       | 0.181807747                        |
| 448       | MinMaxScaler          | RandomForest       | 0.181892752                        |
| 220       | MaxAbsScaler          | RandomForest       | 0.181967743                        |
| 166       | StandardScalerWrapper | ExtremeRandomTrees | 0.182130089                        |
| 194       | StandardScalerWrapper | GradientBoosting   | 0.182191044                        |
| 446       | MinMaxScaler          | GradientBoosting   | 0.182205876                        |
| 460       | MinMaxScaler          | RandomForest       | 0.182306616                        |
| 77        | StandardScalerWrapper | RandomForest       | 0.182756477                        |
| 67        | StandardScalerWrapper | ExtremeRandomTrees | 0.182840651                        |
| 3         | StandardScalerWrapper | RandomForest       | 0.182953506                        |
| 497       | MinMaxScaler          | RandomForest       | 0.183203312                        |
| 106       | RobustScaler          | RandomForest       | 0.183222424                        |
| 495       | StandardScalerWrapper | RandomForest       | 0.183545871                        |
| 113       | StandardScalerWrapper | RandomForest       | 0.183684643                        |
| 58        | StandardScalerWrapper | RandomForest       | 0.1837037                          |
| 135       | MaxAbsScaler          | RandomForest       | 0.183814066                        |
| 420       | MaxAbsScaler          | RandomForest       | 0.183858358                        |
| 101       | StandardScalerWrapper | ExtremeRandomTrees | 0.18392998                         |
| 459       | RobustScaler          | RandomForest       | 0.183957374                        |
| 337       | MinMaxScaler          | ExtremeRandomTrees | 0.183983511                        |
| 170       | SparseNormalizer      | RandomForest       | 0.184022677                        |
| 227       | StandardScalerWrapper | ExtremeRandomTrees | 0.184109421                        |
| 444       | MinMaxScaler          | ExtremeRandomTrees | 0.184199864                        |
| 191       | RobustScaler          | RandomForest       | 0.184254499                        |
| 434       | StandardScalerWrapper | RandomForest       | 0.184298507                        |
| 410       | StandardScalerWrapper | RandomForest       | 0.184377625                        |
| 328       | StandardScalerWrapper | ExtremeRandomTrees | 0.184560212                        |
| 108       | MaxAbsScaler          | ExtremeRandomTrees | 0.184592308                        |
| 301       | StandardScalerWrapper | RandomForest       | 0.184695991                        |
| 451       | MinMaxScaler          | ExtremeRandomTrees | 0.184705201                        |
| 103       | RobustScaler          | LightGBM           | 0.184852621                        |
| 132       | MaxAbsScaler          | GradientBoosting   | 0.184857528                        |
| 27        | StandardScalerWrapper | ExtremeRandomTrees | 0.185017986                        |
| 62        | StandardScalerWrapper | ExtremeRandomTrees | 0.185083973                        |
| 116       | MaxAbsScaler          | GradientBoosting   | 0.185128156                        |
| 353       | MinMaxScaler          | ExtremeRandomTrees | 0.185161979                        |
| 37        | StandardScalerWrapper | ExtremeRandomTrees | 0.185174195                        |
| 411       | StandardScalerWrapper | RandomForest       | 0.185198722                        |
| 17        | StandardScalerWrapper | LightGBM           | 0.185290556                        |
| 91        | MaxAbsScaler          | ExtremeRandomTrees | 0.18545762                         |
| 204       | StandardScalerWrapper | ExtremeRandomTrees | 0.185461197                        |
| 182       | StandardScalerWrapper | ExtremeRandomTrees | 0.185778489                        |
| 39        | StandardScalerWrapper | ExtremeRandomTrees | 0.185979868                        |
| 474       | StandardScalerWrapper | ExtremeRandomTrees | 0.18602163                         |
| 439       | StandardScalerWrapper | ExtremeRandomTrees | 0.186099541                        |
| 452       | RobustScaler          | ExtremeRandomTrees | 0.186103613                        |
| 178       | MinMaxScaler          | ExtremeRandomTrees | 0.186110936                        |
| 334       | SparseNormalizer      | ExtremeRandomTrees | 0.186179484                        |
| 197       | StandardScalerWrapper | RandomForest       | 0.186191545                        |
| 174       | MinMaxScaler          | ExtremeRandomTrees | 0.186448417                        |
| 331       | RobustScaler          | ExtremeRandomTrees | 0.186688484                        |
| 25        | MaxAbsScaler          | ElasticNet         | 0.18680528                         |
| 79        | MinMaxScaler          | ExtremeRandomTrees | 0.186834916                        |
| 491       | StandardScalerWrapper | ExtremeRandomTrees | 0.187016966                        |
| 205       | StandardScalerWrapper | RandomForest       | 0.187057113                        |
| 314       | MinMaxScaler          | ExtremeRandomTrees | 0.187176735                        |
| 476       | RobustScaler          | ExtremeRandomTrees | 0.187198871                        |
| 381       | RobustScaler          | GradientBoosting   | 0.187296314                        |
| 218       | StandardScalerWrapper | GradientBoosting   | 0.187305596                        |
| 145       | MinMaxScaler          | RandomForest       | 0.18736367                         |
| 436       | RobustScaler          | LightGBM           | 0.18736767                         |
| 488       | MaxAbsScaler          | ExtremeRandomTrees | 0.18737676                         |
| 356       | MinMaxScaler          | ExtremeRandomTrees | 0.187499704                        |
| 126       | MinMaxScaler          | LightGBM           | 0.18752192                         |
| 214       | MaxAbsScaler          | GradientBoosting   | 0.187766633                        |
| 305       | MinMaxScaler          | RandomForest       | 0.187980481                        |
| 492       | StandardScalerWrapper | LightGBM           | 0.188146348                        |
| 264       | StandardScalerWrapper | ExtremeRandomTrees | 0.188398363                        |
| 367       | StandardScalerWrapper | RandomForest       | 0.188517146                        |
| 480       | SparseNormalizer      | RandomForest       | 0.188549781                        |
| 349       | RobustScaler          | ExtremeRandomTrees | 0.188562296                        |
| 308       | StandardScalerWrapper | ExtremeRandomTrees | 0.188748603                        |
| 122       | RobustScaler          | ExtremeRandomTrees | 0.188938278                        |
| 376       | MaxAbsScaler          | ExtremeRandomTrees | 0.189136382                        |
| 273       | MaxAbsScaler          | ExtremeRandomTrees | 0.189482366                        |
| 189       | RobustScaler          | ExtremeRandomTrees | 0.189540998                        |
| 304       | MaxAbsScaler          | ExtremeRandomTrees | 0.189595172                        |
| 274       | StandardScalerWrapper | GradientBoosting   | 0.189596277                        |
| 74        | MaxAbsScaler          | ExtremeRandomTrees | 0.189679123                        |
| 276       | MinMaxScaler          | ExtremeRandomTrees | 0.189690955                        |
| 435       | SparseNormalizer      | RandomForest       | 0.18978274                         |
| 329       | StandardScalerWrapper | ExtremeRandomTrees | 0.189997904                        |
| 391       | StandardScalerWrapper | ExtremeRandomTrees | 0.190006797                        |
| 248       | MinMaxScaler          | ExtremeRandomTrees | 0.190313116                        |
| 41        | MaxAbsScaler          | ExtremeRandomTrees | 0.190364881                        |
| 412       | MinMaxScaler          | GradientBoosting   | 0.190436705                        |
| 243       | RobustScaler          | RandomForest       | 0.190605952                        |
| 257       | MinMaxScaler          | RandomForest       | 0.190714921                        |
| 272       | MinMaxScaler          | RandomForest       | 0.190793936                        |
| 129       | StandardScalerWrapper | GradientBoosting   | 0.190979994                        |
| 464       | MinMaxScaler          | ExtremeRandomTrees | 0.191070938                        |
| 28        | MaxAbsScaler          | ExtremeRandomTrees | 0.191122537                        |
| 254       | SparseNormalizer      | GradientBoosting   | 0.191329849                        |
| 299       | MinMaxScaler          | ExtremeRandomTrees | 0.191417043                        |
| 421       | StandardScalerWrapper | GradientBoosting   | 0.192179948                        |
| 371       | MinMaxScaler          | RandomForest       | 0.192392426                        |
| 258       | MinMaxScaler          | RandomForest       | 0.192411123                        |
| 360       | StandardScalerWrapper | RandomForest       | 0.192436778                        |
| 489       | SparseNormalizer      | LightGBM           | 0.192641881                        |
| 8         | MaxAbsScaler          | DecisionTree       | 0.193020019                        |
| 44        | StandardScalerWrapper | GradientBoosting   | 0.193363433                        |
| 167       | RobustScaler          | LightGBM           | 0.193871403                        |
| 261       | MaxAbsScaler          | GradientBoosting   | 0.195117539                        |
| 324       | StandardScalerWrapper | ExtremeRandomTrees | 0.19583183                         |
| 10        | StandardScalerWrapper | DecisionTree       | 0.196590878                        |
| 414       | MaxAbsScaler          | ExtremeRandomTrees | 0.197089097                        |
| 357       | StandardScalerWrapper | ExtremeRandomTrees | 0.197132361                        |
| 12        | MinMaxScaler          | ExtremeRandomTrees | 0.197214567                        |
| 85        | StandardScalerWrapper | RandomForest       | 0.19751637                         |
| 52        | StandardScalerWrapper | ExtremeRandomTrees | 0.197834508                        |
| 285       | SparseNormalizer      | RandomForest       | 0.19797267                         |
| 7         | MaxAbsScaler          | RandomForest       | 0.198537452                        |
| 277       | StandardScalerWrapper | ExtremeRandomTrees | 0.199833488                        |
| 481       | StandardScalerWrapper | ExtremeRandomTrees | 0.199847538                        |
| 354       | RobustScaler          | ExtremeRandomTrees | 0.203411818                        |
| 155       | StandardScalerWrapper | RandomForest       | 0.203652532                        |
| 250       | MaxAbsScaler          | RandomForest       | 0.204483954                        |
| 14        | StandardScalerWrapper | ElasticNet         | 0.204714308                        |
| 231       | MinMaxScaler          | ExtremeRandomTrees | 0.205107758                        |
| 86        | StandardScalerWrapper | ExtremeRandomTrees | 0.206573894                        |
| 186       | StandardScalerWrapper | DecisionTree       | 0.20802764                         |
| 366       | MinMaxScaler          | DecisionTree       | 0.209525874                        |
| 232       | RobustScaler          | ExtremeRandomTrees | 0.209941397                        |
| 121       | StandardScalerWrapper | GradientBoosting   | 0.210084717                        |
| 343       | MinMaxScaler          | GradientBoosting   | 0.21036495                         |
| 2         | StandardScalerWrapper | ElasticNet         | 0.212042108                        |
| 6         | MinMaxScaler          | DecisionTree       | 0.212432524                        |
| 163       | MinMaxScaler          | RandomForest       | 0.214360431                        |
| 19        | StandardScalerWrapper | ElasticNet         | 0.216964702                        |
| 228       | RobustScaler          | ExtremeRandomTrees | 0.21968984                         |
| 20        | TruncatedSVDWrapper   | ElasticNet         | 0.220103076                        |
| 15        | StandardScalerWrapper | ElasticNet         | 0.221044137                        |
| 259       | SparseNormalizer      | DecisionTree       | 0.222488814                        |
| 22        | TruncatedSVDWrapper   | ElasticNet         | 0.222790169                        |
| 18        | MinMaxScaler          | DecisionTree       | 0.228681804                        |
| 173       | MinMaxScaler          | DecisionTree       | 0.230437233                        |
| 237       | MinMaxScaler          | SGD                | 0.231384017                        |
| 402       | StandardScalerWrapper | DecisionTree       | 0.232003166                        |
| 322       | RobustScaler          | DecisionTree       | 0.232166047                        |
| 187       | MinMaxScaler          | DecisionTree       | 0.232705011                        |
| 13        | StandardScalerWrapper | ElasticNet         | 0.235524912                        |
| 49        | StandardScalerWrapper | DecisionTree       | 0.237119485                        |
| 92        | RobustScaler          | DecisionTree       | 0.241947998                        |
| 9         | MinMaxScaler          | DecisionTree       | 0.253827356                        |
| 102       | MinMaxScaler          | DecisionTree       | 0.256963716                        |
| 5         | StandardScalerWrapper | LassoLars          | 0.260923169                        |
| 198       | RobustScaler          | DecisionTree       | 0.279708013                        |
| 416       | MaxAbsScaler          | DecisionTree       | 0.285545245                        |
| 316       | StandardScalerWrapper | KNN                | nan                                |

## Subchallenge 2

__Topic:__ Utilizing a previously published _in vivo_ transcription data set from [Mok et al,. Science 2015](https://www.ncbi.nlm.nih.gov/pubmed/25502316), predict the resistance status of malaria isolates utilizing _in vitro_ transcription data.

__Data:__ _in vivo_ transcription data set ([Mok et al,. Science 2015](https://www.ncbi.nlm.nih.gov/pubmed/25502316)) to predict the parasite clearance rate of malaria parasite isolates based on in vitro transcriptional profiles.

|                               | Training Set                                                                            | Test Set                                                                              |
|:-----------------------------:|:---------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------:|
| Dataset                       | Published from [Mok et al,. Science 2015](https://www.ncbi.nlm.nih.gov/pubmed/25502316) | Unpublished                                                                           |
| Number of Isolates            | 1043                                                                                    | 32                                                                                    |
| Isolate Collection Site       | Southeast Asia                                                                          | Thai-Myanmar Border                                                                   |
| Isolate Collection Years      | 2012-2014                                                                               | 2007-2012                                                                             |
| Sample Type                   | _in vivo_                                                                               | _in vitro_                                                                            |
| Synchronized?                 | Not Synchronized                                                                        | Synchronized                                                                          |
| Number of Samples per Isolate | 1                                                                                       | 8                                                                                     |
| Additional Attributes         | ~18 Hours Post Invasion (hpi), Non-perturbed, No replicates                             | Two time points (6 hpi and 24 hpi), Perturbed and Non-perturbed, Biological Replicate |

### Subchallenge 2 Final Submission

**Data Preprocessing:** None, except removal of certain attributes. (`Country` - test data only contains "Thailand_Myanmar_Border", which isn't in the training data, `Kmeans_Grp` - not included in the testing data, and `Asexual_stage__hpi_` - testing data has different stages.)

- [Data Shaping Script (PySpark on Apache Spark)](https://github.com/colbyford/malaria_DREAM2019/blob/master/SubChallenge2/scripts/SubChallenge2_DataPrep.py)
- [Vectorized Training Data (Independent Variables)](https://github.com/colbyford/malaria_DREAM2019/blob/master/SubChallenge2/data/sc2_X_train.pkl)
- [Vectorized Training Data (Dependent Variable)](https://github.com/colbyford/malaria_DREAM2019/blob/master/SubChallenge2/data/sc2_y_train.pkl)

**Training Parameters:** The following are the training parameters for the Azure Machine Learning Service:

| Parameter                   | Value          |
|:---------------------------:|:--------------:|
| Task                        | Classification |
| Number of Iterations        | 100            |
| Iteration Timeout (minutes) | 20             |
| Max Cores per Iteration     | 14             |
| Primary Metric              | Weighted AUC   |
| Preprocess Data?            | True           |
| k-Fold Cross-Validations    | 10 folds       |

- [Submission Script for SubChallenge 2 Data to AMLS](https://github.com/colbyford/malaria_DREAM2019/blob/master/SubChallenge2/scripts/Submit_SubCh2_to_AMLS.py)

**Model:** Voting Ensemble model (using soft voting) of 98 previous models (including models trained using Logistic Regression, SVM, Gradient Boosting, SGD, Naïve Bayes, ~~KNN~~, Random Forests, and Extreme Random Trees), each having been trained on pre-processed data using various steps (including imputation, normalization, and scaling).

- [Ensemble Model Pickle File](https://github.com/colbyford/malaria_DREAM2019/blob/master/SubChallenge2/model/amls_model_7-31-19/sc2_model.pkl)

### Model Information

**Model Date:** 7/31/2019

**Platform:** Azure Machine Learning Service

**Best Model:** `VotingEnsemble`

**Model Description:** Using previous iterations of the machine learning model training, a SoftVoting Ensemble model was created. This uses the average of the class probabilities of previous iterations.

| Metric                           | Accuracy |
|:--------------------------------:|:--------:|
| f1_score_macro                   | 0.6084   |
| AUC_micro                        | 0.9445   |
| AUC_macro                        | 0.8475   |
| recall_score_micro               | 0.8101   |
| recall_score_weighted            | 0.8101   |
| average_precision_score_weighted | 0.8707   |
| weighted_accuracy                | 0.8585   |
| precision_score_macro            | 0.6217   |
| precision_score_micro            | 0.8101   |
| balanced_accuracy                | 0.6027   |
| log_loss                         | 0.4455   |
| recall_score_macro               | 0.6027   |
| precision_score_weighted         | 0.8      |
| AUC_weighted                     | 0.8705   |
| average_precision_score_micro    | 0.8911   |
| f1_score_weighted                | 0.8019   |
| f1_score_micro                   | 0.8101   |
| norm_macro_recall                | 0.354    |
| average_precision_score_macro    | 0.7344   |
| accuracy                         | 0.8101   |

## Confusion Matrix

| True\Pred | 0   | 1   | 2   |
| --------- | --- | --- | --- |
| **0**     | 661 | 74  | 0   |
| **1**     | 115 | 184 | 0   |
| **2**     | 6   | 3   | 0   |

## All Previous Iterations:

| ITERATION | RUN_PREPROCESSOR      | RUN_ALGORITHM      | AUC_WEIGHTED |
|:---------:|:---------------------:|:------------------:|:------------:|
| 98        |                       | VotingEnsemble     | 0.870471056  |
| 99        |                       | StackEnsemble      | 0.865215516  |
| 65        | StandardScalerWrapper | LogisticRegression | 0.86062304   |
| 33        | StandardScalerWrapper | LogisticRegression | 0.859881677  |
| 97        | StandardScalerWrapper | LogisticRegression | 0.858791006  |
| 44        | StandardScalerWrapper | LogisticRegression | 0.856105491  |
| 73        | StandardScalerWrapper | LogisticRegression | 0.855502817  |
| 17        | RobustScaler          | SVM                | 0.855452622  |
| 43        | StandardScalerWrapper | LogisticRegression | 0.855368394  |
| 61        | RobustScaler          | LogisticRegression | 0.854357599  |
| 21        | RobustScaler          | LogisticRegression | 0.854352919  |
| 50        | StandardScalerWrapper | LogisticRegression | 0.854271919  |
| 76        | RobustScaler          | LogisticRegression | 0.854138203  |
| 27        | RobustScaler          | LogisticRegression | 0.853683531  |
| 88        | StandardScalerWrapper | LogisticRegression | 0.853545166  |
| 26        | RobustScaler          | LogisticRegression | 0.852415532  |
| 37        | TruncatedSVDWrapper   | LogisticRegression | 0.85221177   |
| 81        | StandardScalerWrapper | LogisticRegression | 0.851581173  |
| 45        | StandardScalerWrapper | LogisticRegression | 0.851506849  |
| 54        | RobustScaler          | LogisticRegression | 0.851398541  |
| 82        | TruncatedSVDWrapper   | LogisticRegression | 0.851126263  |
| 72        | RobustScaler          | LogisticRegression | 0.851049949  |
| 79        | StandardScalerWrapper | LogisticRegression | 0.851018179  |
| 86        | RobustScaler          | LogisticRegression | 0.850750784  |
| 71        | StandardScalerWrapper | LogisticRegression | 0.850640655  |
| 66        | SparseNormalizer      | LogisticRegression | 0.849491634  |
| 92        | RobustScaler          | LogisticRegression | 0.84938126   |
| 24        | RobustScaler          | LogisticRegression | 0.848420725  |
| 84        | RobustScaler          | LinearSVM          | 0.848152508  |
| 56        | RobustScaler          | LogisticRegression | 0.848021306  |
| 75        | RobustScaler          | LogisticRegression | 0.847751321  |
| 57        | MaxAbsScaler          | LogisticRegression | 0.845523427  |
| 95        | StandardScalerWrapper | LogisticRegression | 0.844397139  |
| 53        | StandardScalerWrapper | LogisticRegression | 0.844102598  |
| 67        | TruncatedSVDWrapper   | LogisticRegression | 0.843800432  |
| 74        | SparseNormalizer      | LogisticRegression | 0.843712149  |
| 22        | MaxAbsScaler          | LogisticRegression | 0.843654091  |
| 83        | TruncatedSVDWrapper   | LogisticRegression | 0.843541572  |
| 49        | MaxAbsScaler          | LogisticRegression | 0.843507898  |
| 80        | StandardScalerWrapper | LogisticRegression | 0.843064786  |
| 18        | StandardScalerWrapper | LogisticRegression | 0.843022237  |
| 47        | RobustScaler          | LightGBM           | 0.842506023  |
| 29        | RobustScaler          | LinearSVM          | 0.841886734  |
| 59        | TruncatedSVDWrapper   | LogisticRegression | 0.840557865  |
| 32        | RobustScaler          | LinearSVM          | 0.840290047  |
| 60        | PCA                   | LogisticRegression | 0.839967305  |
| 31        | MaxAbsScaler          | GradientBoosting   | 0.838920218  |
| 51        | StandardScalerWrapper | LogisticRegression | 0.837486895  |
| 85        | TruncatedSVDWrapper   | LogisticRegression | 0.837383394  |
| 62        | StandardScalerWrapper | LogisticRegression | 0.837110668  |
| 87        | StandardScalerWrapper | LogisticRegression | 0.837097803  |
| 90        | TruncatedSVDWrapper   | LogisticRegression | 0.836835098  |
| 68        | MinMaxScaler          | LogisticRegression | 0.836478087  |
| 34        | TruncatedSVDWrapper   | LogisticRegression | 0.835800612  |
| 12        | MinMaxScaler          | LightGBM           | 0.835284383  |
| 36        | StandardScalerWrapper | LightGBM           | 0.834583074  |
| 8         | StandardScalerWrapper | SGD                | 0.834314309  |
| 94        | StandardScalerWrapper | LogisticRegression | 0.834179993  |
| 46        | MaxAbsScaler          | LinearSVM          | 0.834053264  |
| 63        | StandardScalerWrapper | LogisticRegression | 0.833803809  |
| 23        | StandardScalerWrapper | LightGBM           | 0.83363186   |
| 58        | MaxAbsScaler          | LinearSVM          | 0.832909387  |
| 96        | StandardScalerWrapper | LogisticRegression | 0.83215075   |
| 78        | MaxAbsScaler          | GradientBoosting   | 0.830390117  |
| 16        | MinMaxScaler          | SVM                | 0.830325182  |
| 0         | StandardScalerWrapper | SGD                | 0.827602503  |
| 41        | MinMaxScaler          | SVM                | 0.826022547  |
| 69        | TruncatedSVDWrapper   | LinearSVM          | 0.825353557  |
| 35        | MaxAbsScaler          | LogisticRegression | 0.823008572  |
| 70        | MinMaxScaler          | LogisticRegression | 0.822648165  |
| 38        | MaxAbsScaler          | LightGBM           | 0.82206555   |
| 64        | RobustScaler          | GradientBoosting   | 0.821621848  |
| 25        | TruncatedSVDWrapper   | SVM                | 0.820205555  |
| 5         | StandardScalerWrapper | LightGBM           | 0.819725881  |
| 2         | MinMaxScaler          | LightGBM           | 0.819564758  |
| 9         | MinMaxScaler          | SGD                | 0.819551222  |
| 30        | RobustScaler          | SVM                | 0.818433136  |
| 55        | MinMaxScaler          | LogisticRegression | 0.817255009  |
| 91        | MinMaxScaler          | LogisticRegression | 0.816844835  |
| 89        | MaxAbsScaler          | SVM                | 0.816578056  |
| 6         | StandardScalerWrapper | SGD                | 0.811688321  |
| 1         | StandardScalerWrapper | SGD                | 0.80559481   |
| 42        | RobustScaler          | GradientBoosting   | 0.802210742  |
| 3         | StandardScalerWrapper | SGD                | 0.800184053  |
| 39        | MaxAbsScaler          | RandomForest       | 0.797587204  |
| 28        | StandardScalerWrapper | ExtremeRandomTrees | 0.793047181  |
| 52        | StandardScalerWrapper | RandomForest       | 0.789606687  |
| 11        | MaxAbsScaler          | LightGBM           | 0.789403722  |
| 4         | StandardScalerWrapper | ExtremeRandomTrees | 0.779719545  |
| 14        | MinMaxScaler          | LightGBM           | 0.777246393  |
| 19        | RobustScaler          | SVM                | 0.762799099  |
| 13        | MinMaxScaler          | ExtremeRandomTrees | 0.751820414  |
| 77        | StandardScalerWrapper | KNN                | 0.743536371  |
| 93        | StandardScalerWrapper | RandomForest       | 0.740452619  |
| 48        | RobustScaler          | KNN                | 0.738236639  |
| 10        | MinMaxScaler          | RandomForest       | 0.738015441  |

---

## Discussion

By using distributed processing of the data preparation, we can successfully shape and manage a malaria dataset of over 40,000 genetic attributes. This is complete with scalable vectorization of the training data, which allowed for many machine learning models to be generated. By tracking the individual performance results of each machine learning model, we can determine which model is most useful. In addition, ensemble modeling of the various singular models proved effective for both subchallenges in this work.

## References

1. NIEHS-NCATS-UNC DREAM Toxicogenetics Challenge ([syn1761567](https://www.synapse.org/#!Synapse:syn1761567))
2. Caruana, R., Niculescu-Mizil, A., Crew, G. & Ksikes, A. Ensemble selection from libraries of models. _In Proceedings of the Twenty-first International Conference on Machine Learning_, ICML ’04, 18–, DOI: 10.1145/1015330.1015432 (ACM,New York, NY, USA, 2004).
