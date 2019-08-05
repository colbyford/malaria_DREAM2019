# Model Information

**Model Date:** 7/31/2019

**Run Id:** `AutoML_43d7399a-3f89-4728-b884-51be811a8e1a_98`

**Package:** Azure Machine Learning Service - AutoML

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

## Precision-Recall Curve

![](Precision-Recall.PNG)

## ROC Curve

![](ROC.PNG)

## Calibration Curve

![](Calibration.PNG)

## Gain Curve

![](Gain.PNG)

## Lift Curve

![](Lift.PNG)

## All Previous Iterations:

| ITERATION |    RUN_PREPROCESSOR   |    RUN_ALGORITHM   | AUC_WEIGHTED |
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
