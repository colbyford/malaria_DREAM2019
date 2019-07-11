# Model Information

**Model Date:** 7/11/2019

**Run Id:** `AutoML_16c66c54-b465-47ef-b703-0ec0eabaf1e0`

**Package:** Azure Machine Learning Service - AutoML

**Best Model:** `VotingEnsemble`

**Model Description:** Using previous iterations of the machine learning model training, a SoftVoting Ensemble model was created. This uses the average of the class probabilities of previous iterations.

| Metric                                 | Value   |
|:--------------------------------------:|:-------:|
| normalized_mean_absolute_error         | 0.148   |
| normalized_root_mean_squared_log_error | 0.212   |
| normalized_median_absolute_error       | 0.1192  |
| mean_absolute_error                    | 0.4094  |
| median_absolute_error                  | 0.3297  |
| root_mean_squared_error                | 0.5177  |
| spearman_correlation                   | 0.1657  |
| root_mean_squared_log_error            | 0.2053  |
| normalized_root_mean_squared_error     | 0.1871  |
| r2_score                               | -0.0958 |
| mean_absolute_percentage_error         | 31.93   |
| explained_variance                     | 0.01555 |

## All Previous Iterations:

| ITERATION | RUN_PREPROCESSOR      | RUN_ALGORITHM      | NORMALIZED_MEAN_ABSOLUTE_ERROR |
|:---------:|:---------------------:|:------------------:|:------------------------------:|
| 23        |                       | VotingEnsemble     | 0.147959247                    |
| 20        | MaxAbsScaler          | ElasticNet         | 0.158748684                    |
| 14        | StandardScalerWrapper | ElasticNet         | 0.159217401                    |
| 8         | StandardScalerWrapper | LassoLars          | 0.159246676                    |
| 18        | StandardScalerWrapper | ElasticNet         | 0.159246676                    |
| 24        |                       | StackEnsemble      | 0.161323162                    |
| 2         | StandardScalerWrapper | RandomForest       | 0.168574008                    |
| 21        | RobustScaler          | RandomForest       | 0.168981097                    |
| 22        | StandardScalerWrapper | RandomForest       | 0.169930561                    |
| 7         | MinMaxScaler          | RandomForest       | 0.182787316                    |
| 3         | StandardScalerWrapper | ExtremeRandomTrees | 0.189436995                    |
| 19        | TruncatedSVDWrapper   | ElasticNet         | 0.205071315                    |
| 12        | StandardScalerWrapper | ElasticNet         | 0.207441513                    |
| 1         | StandardScalerWrapper | ElasticNet         | 0.208125798                    |
| 11        | RobustScaler          | DecisionTree       | 0.209585427                    |
| 5         | MinMaxScaler          | DecisionTree       | 0.210622764                    |
| 13        | StandardScalerWrapper | ElasticNet         | 0.213650879                    |
| 0         | StandardScalerWrapper | ElasticNet         | 0.216105211                    |
| 4         | StandardScalerWrapper | DecisionTree       | 0.226021373                    |
| 10        | MinMaxScaler          | DecisionTree       | 0.226357514                    |
| 15        | StandardScalerWrapper | DecisionTree       | 0.228623752                    |
| 6         | RobustScaler          | LassoLars          | 0.231107756                    |
| 17        | MinMaxScaler          | DecisionTree       | 0.233831721                    |
| 9         | StandardScalerWrapper | DecisionTree       | 0.248281346                    |
| 16        | MaxAbsScaler          | DecisionTree       | 0.259931808                    |
