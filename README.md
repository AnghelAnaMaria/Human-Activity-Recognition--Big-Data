# Human Activity Recognition

This project identifies human activities from smartphone sensor data. It uses Apache Spark (PySpark and MLlib) and TensorFlow.

The full code is in one notebook: [405_Anghel_Ana-Maria_Cod-sursa.ipynb](./405_Anghel_Ana-Maria_Cod-sursa.ipynb).

## 1. Purpose

The project does these tasks:

- It analyzes the dataset with the Spark DataFrame API and Spark SQL.
- It removes features that are redundant or that are not useful.
- It trains and compares three Spark MLlib classifiers and one neural network.
- It tunes the hyperparameters of each MLlib classifier.
- It shows real-time predictions with Spark Structured Streaming.

## 2. Dataset

The dataset is "Human Activity Recognition Using Smartphones" (HAR). The University of California created it.

- Source: [Kaggle - Human Activity Recognition with Smartphones](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones)
- Thirty volunteers wore a smartphone on the waist.
- The accelerometer and the gyroscope of the smartphone recorded the signals at 50 Hz.
- The signals are divided into time windows of 2.56 seconds, with an overlap of 50%.
- Each row is a summary of one time window. It is not one instantaneous measurement.
- Each row has 561 numerical features, a subject ID and an activity label.

The dataset has six activity classes:

| Class              | Type    |
| ------------------ | ------- |
| WALKING            | Dynamic |
| WALKING_UPSTAIRS   | Dynamic |
| WALKING_DOWNSTAIRS | Dynamic |
| SITTING            | Static  |
| STANDING           | Static  |
| LAYING             | Static  |

The dataset has two files:

| File        | Rows  |
| ----------- | ----- |
| `train.csv` | 7,352 |
| `test.csv`  | 2,947 |

## 3. Requirements

The notebook runs in Google Colab. Spark runs in local mode (`local[*]`) on two CPU cores.

You must have these Python libraries:

- `pyspark`
- `tensorflow`
- `scikit-learn`
- `pandas`
- `numpy`
- `matplotlib`

## 4. How to run the notebook

1. Download `train.csv` and `test.csv` from Kaggle.
2. Put the two files in this Google Drive folder: `MyDrive/big_data_dataset/`.
3. Open the notebook in Google Colab.
4. Run the cells in sequence, from the top to the bottom.
5. When Colab asks for access to Google Drive, give the permission.

If you keep the files in a different location, change the paths in the cell that reads the data.

## 5. Exploratory data analysis

The notebook does each analysis two times: one time with the DataFrame API and one time with Spark SQL.

The analysis gives these results:

- The training data has no missing values.
- The training data has no duplicate rows.
- The feature values are normalized to the range [-1, 1].
- The six classes have almost the same number of rows. Thus, the project does not use oversampling or undersampling.
- Each subject has a different number of rows for each activity. Subject 1 has more WALKING rows than the other subjects. The project keeps these rows, because the training data has only 21 subjects.
- Two LAYING rows have the extreme values -1.0 and 1.0 for `tBodyAcc-mean()-X`.

## 6. Data preparation

### 6.1 Feature selection

The project removes features in three steps:

| Step                      | Rule                                                                                    | Features removed | Features kept |
| ------------------------- | --------------------------------------------------------------------------------------- | ---------------- | ------------- |
| 1. Correlation filter     | Remove a feature if its Pearson correlation with a different feature is more than 0.95. | 308              | 253           |
| 2. Discrimination filter  | Remove the 5% of features with the smallest difference between activity means.          | 13               | -             |
| 3. Subject leakage filter | Remove a feature if its mean changes more between subjects than between activities.     | 66               | 183           |

Spark calculates the correlation matrix in a distributed operation.

### 6.2 Pipeline

All the MLlib models use the same Spark ML pipeline:

1. `VectorAssembler` puts the 183 features into one vector.
2. `StandardScaler` sets the mean of each feature to 0 and the standard deviation to 1.
3. `PCA` decreases the vector to 75 components. These components keep 93.3% of the variance.
4. `StringIndexer` changes the activity name into a numerical label.
5. The classifier makes the prediction.

## 7. Models

### 7.1 Spark MLlib classifiers

The project trains three classifiers:

- **Multinomial Logistic Regression.** This model is applicable to multiclass problems with continuous numerical features.
- **Decision Tree.** This model can find non-linear relations between the features and the classes.
- **Random Forest.** This model combines many decision trees. It has a lower risk of overfitting than one decision tree.

### 7.2 Hyperparameter tuning

The project divides the training data by subject, not by row. Thus, the validation data contains only subjects that the model did not see during training. A random split by row can put data from the same subject in the training data and in the validation data. Then the model can learn the movement style of each person.

| Model               | Method                                          | Parameters tested                                           | Best parameters                             |
| ------------------- | ----------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------- |
| Logistic Regression | Grid search, one 80/20 split by subject         | `regParam`: 0.001, 0.01, 0.1. `maxIter`: 50, 100, 200.      | `regParam` = 0.001, `maxIter` = 50          |
| Decision Tree       | Grid search, one 80/20 split by subject         | `maxDepth`: 5, 10, 15, 20. `minInstancesPerNode`: 1, 5, 10. | `maxDepth` = 15, `minInstancesPerNode` = 10 |
| Random Forest       | Grid search, 5-fold cross-validation by subject | `numTrees`: 50, 100, 200. `maxDepth`: 5, 10, 15.            | `numTrees` = 200, `maxDepth` = 15           |

The Spark `CrossValidator` divides rows at random. Thus, the project uses its own cross-validation function for the Random Forest.

### 7.3 Neural network (TensorFlow)

The project also trains a Multilayer Perceptron (MLP). The data is tabular and has no spatial or temporal structure. Thus, the project does not use a CNN or an RNN.

- Input: the 75 PCA components from the Spark pipeline.
- Hidden layers: 128, 64 and 32 neurons, with the ReLU activation.
- Output layer: 6 neurons, with the softmax activation.
- Optimizer: Adam.
- Loss function: sparse categorical cross-entropy.
- Validation: 20% of the training data.
- Batch size: 32.
- Early stopping: it monitors the validation accuracy, with a patience of 3 epochs. It keeps the best weights.

The training stopped after 5 epochs. The best validation accuracy was 93.5%, at epoch 2.

## 8. Results

All values are for the test data (2,947 rows).

| Model                           | Accuracy   | F1 score        |
| ------------------------------- | ---------- | --------------- |
| Logistic Regression (default)   | 0.9162     | 0.9160          |
| **Logistic Regression (tuned)** | **0.9213** | -               |
| Decision Tree (default)         | 0.7696     | 0.7694          |
| Decision Tree (tuned)           | 0.7750     | -               |
| Random Forest (default)         | 0.8616     | 0.8598          |
| Random Forest (tuned)           | 0.8829     | -               |
| MLP (TensorFlow)                | 0.8975     | 0.90 (weighted) |

The tuned Logistic Regression gives the best accuracy.

All models make most errors in two areas:

- They mix SITTING and STANDING.
- They mix WALKING, WALKING_UPSTAIRS and WALKING_DOWNSTAIRS.

All models identify LAYING almost without errors.

## 9. Static and dynamic activities (UDF)

The project uses user-defined functions (UDFs) to put the six activities into two groups: "Static" and "Moving". This test examines the predictions of the Logistic Regression model.

| Actual group | Predicted as Moving | Predicted as Static |
| ------------ | ------------------- | ------------------- |
| Moving       | 1,376               | 11                  |
| Static       | 0                   | 1,560               |

The area under the ROC curve (AUC) is 0.99995. Thus, the model separates static activities from dynamic activities almost perfectly. The model did not have specific training for this task.

## 10. Spark Structured Streaming

The project shows real-time predictions with Spark Structured Streaming:

1. The notebook divides the test data into 30 CSV files of 100 rows each.
2. Spark monitors an input folder.
3. The notebook copies the files into the input folder, one file at a time.
4. For each new batch, the Logistic Regression pipeline makes predictions.
5. Spark writes the predictions and the metrics (accuracy and F1 score) to CSV files.

The notebook sent two batches. These are the results:

| Batch | Accuracy | F1 score |
| ----- | -------- | -------- |
| 0     | 0.91     | 0.909    |
| 1     | 0.93     | 0.930    |
