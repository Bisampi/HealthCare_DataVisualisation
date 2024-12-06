This project involves analyzing healthcare-related data to understand key insights and apply machine learning algorithms for prediction tasks. The workflow includes:

Data Analysis: 
Exploring correlations between admission types, medical conditions, and billing amounts.
Feature Engineering: Preprocessing data and handling class imbalance.
Modeling: Comparing the performance of Random Forest and XGBoost classifiers.
Key Features:
Correlation analysis with heatmaps.
Grouped bar plots for billing amount trends.
Hyperparameter tuning for machine learning models.
Handling imbalanced datasets using techniques like scale_pos_weight.
Technologies Used:
Python: Programming language.
Libraries:
pandas, numpy: Data manipulation and analysis.
matplotlib, seaborn: Data visualization.
scikit-learn: Machine learning.
xgboost: Gradient boosting.
For the Exploratory Data analysis I have used python libraries then I divided the data set in test and train for further analysis.
SMOTE (Synthetic Minority Over-sampling Technique) : SMOTE is a popular technique used to address class imbalance by generating synthetic samples for the minority class. 
This is particularly useful in classification problems where one class (e.g., patients with a certain condition) has fewer examples than the other class (e.g., patients without the condition), 
which can lead to biased models.

The machine models I have used for train the data set are:

Random Forest
Random Forest is an ensemble learning method that combines multiple decision trees to improve predictive accuracy and control overfitting.
Each tree is trained on a random subset of the dataset, and predictions are made by averaging the outcomes (for regression) or by majority vote (for classification).

How it Worked Here
The Random Forest model achieved a 63% accuracy on the test data.
Strengths:
It is robust to noise and overfitting due to its ensemble nature.
The model handled missing values and nonlinear data relationships effectively.

XGBoost (Extreme Gradient Boosting)
XGBoost is a powerful gradient boosting algorithm that optimizes decision trees sequentially to minimize errors. It incorporates regularization to prevent overfitting and allows for parallelization, making it faster and more efficient than traditional gradient boosting methods.

How it Worked Here
After hyperparameter tuning, the XGBoost model achieved 59.59% accuracy.
Strengths:
Handles missing data internally and offers fine-tuning options for imbalanced datasets using scale_pos_weight.
Better flexibility in adjusting hyperparameters like learning_rate, max_depth, and gamma.


Results Table
Below is a summary of the performance of different models in the project:

Metric	Random Forest	XGBoost
Best Parameters	n_estimators: 300, max_depth: 15	n_estimators: 300, max_depth: 9, learning_rate: 0.01, subsample: 0.6
Test Accuracy	       63.00%	   59.59%
Precision (Class 0)	  0.60	   0.60
Recall (Class 0)	    1.00	   0.99
Precision (Class 1)	  0.88	   0.63
Recall (Class 1)	    0.03	   0.02
Weighted F1-Score	    0.47	   0.46

