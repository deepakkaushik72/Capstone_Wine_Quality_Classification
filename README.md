### CapstoneProject : Wine Classification (Quality Ratings - 1 through 10)
#### Link to Jupyter Notebook: https://github.com/deepakkaushik72/Capstone_Wine_Quality_Classification/blob/main/red_white_classification.ipynb
#### https://github.com/deepakkaushik72/Capstone_Wine_Quality_Classification/blob/main/red_white_classification%20ANN.ipynb
#### Link to Images: https://github.com/deepakkaushik72/Capstone_Wine_Quality_Classification/tree/main/Images


#### 1. BUSINESS PROBLEM / OBJECTIVE:
> - The question that I am trying to answer is to build a classifier to correctly classify the wine on a scale of 1 to 10 (1 - lowest quality and 10 – highest quality) basis the Physical characteristics and the Chemical composition of the wine, leveraging different supervised machine learning classification algorithms.
> - 1. Classsify the Wine Samples of White and Red wine on a scale of 1 to 10 and achieving a accuracy of over ~ 65%
> - 2. Recall of over 70% for class 5 and class 6
> - 3. Identify the features that influence the quality of wine the most
#### 2. Data Source: Kaggle - https://www.kaggle.com/datasets/mihaltursukov/wine-quality/dataLinks to an external site.
> - 1. 6497 wine data points: 1599 for Red Wine with Quality ratings as 3, 4, 5, 6, 7, 8 and 4898 for White Wine with Quality ratings 3, 4, 5, 6, 7, 8, 9
> - 2. 12 features/physical characteristics: ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 
   'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'color']
> - 3. Output variable: “Quality” (1 through 10)

> Following are the Supervised Machine Learning algorithms that will be used to categorize the Wine quality and select the best 
basis the accuracy, Recall and interpretability:
> - 1. Logistics Regression 2. KNN 3. Decision Trees 4. Support Vector Machines 5. Random Forest 6. Gradient Boosting (XGBoost) 7. Neural Networks
      
#### 3. SUCCESS METRICS:
> - 1. Accuracy score of Wine classification: Accuracy > 65%, Recall, Precision > 65%
> - 2. Highest Recall for Class 5 and 6 (over 70% for class 5 and class 6)
> - 3. Optimizing the F1 Score (Precision-Recall Tradeoff)
> - 4. Clear communication of the features that influence the Quality of Wine the most
#### 4. DATA PREPARATION:
> - There are 6497 records in the RED and WHITE wine csv files: 1599 for Red Wine and 4898 for White Wine
> - There are no duplicate records in the dataset
> - Its a Multiclass Classification problem (Classification into Quality ratings: 1 through 10)
> - **Imbalanced class distribution as there are only 1% to 3% in Quality 3,4, 8 and 9** 
> - Looked at 12 features (11 numerical and 1 categorical which is "Color")
> - Descriptive Statistics: Wide range of values across various features. Requires scaling of the data for Classification models like KNN, SVM, Logistic Regression
> - Combined the Red and White Dataset into one Data set called "Wine" with White Wine labelled as "0" and Red as "1"
#### 5. EXPLORATORATORY DATA ANALYSIS (EDA)
> Following are the Exploratory Data Analysis techniques used:
> - Histogram and KDE Plot for each of the features
> - Box Plot of the Target Variable (Quality of White and Red Wine)
> - Heatmap to find the correlation between the Features that may show linear relationship
> - p-Values for Red, White and Combined dataset to find the features that may be less statistically significant
> - Line plot for each of the features with mean values across different Quality ratings to find the linear trend/relationship
> - PairPlot to find some linear and nonlinear relationships between various featues and Quality ratings
#### 6. EDA FINDINGS
> - Quality of Wine improves as the Volatile acidity decreases for RED wine.
> - Quality of wine improves as the Citric acid increases for RED wine.
> - Alcohol content seems to have a strong correlation with the quality of wine for both RED and WHITE wines.
> - Sulphates also seems to have a positive correlation with the quality of wine for RED Wines
> - Quality of RED wine seems to go up with decrease in pH but vice versa for White Wine
> - Chrolides seems to have a negative correlation with the quality of wine for both RED and WHITE wines
> - Total Sulfur dioxide and Free Sulfur dioxide seems folows similar trend with Quality but not very significant correlation
> - It appears that certain features like Alcohol has a strong positive correlation with wine quality.
> - Alcohol and density seems to have a negative multicollinearity issue.
> - Total Sulfur Dioxide and Free Sulfur Dioxide also seems to have a positive multicollinearity issue
> - WHITE WINE:Citric Acid & Free Sulfur Dioxide seems to be statistically less significant (p-Values > 0.5)
> - RED WINE: Residual Sugar seems to be statistically less significant(p-Values > 0.5)
#### 7. FEATURE SELECTION AND ENGINEERING:
> - There are **12 Features Identified** for building the ML models: 'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'color']. There are both Linear and Non-Linear features that nneds to be included
> - Target Variable is "Quality" (1 through 10)
> - However basis the EDA analysis and domain knowledge, I will be using **only 9 of the 11 features** for model building and selection.
> - All numerical Features **excluding the "Density" feature** as it has a strong multicollinearity with Alcohol.
> - Only one of the Sulfur Dioxide features (**Free Sulfur Dioxide**) is used in Modelling as it has a strong Multi collinearity with **Total Sulfur Dioxide (Excluded)**.
> - **Only 10 features Included**
> - Color feature will be one-hot encoded and seems to be important for Quality ratings.
> - Quality will be the target variable.
> - Since its an imbalanced DataSet where Quality ratings samples of 3, 4, 8 and 9 range from 1%-3%. **SMOTE technique for Oversampling** is used to create a balanced Dataset
> - Created a preprocess for Scaling 8 Numerical features and OneHotEncoding for 1 categorical features i.e "Color"
> - Created a Train and Test Split of the Input features and Target Variable (Applied 20% test split and Stratify = y)
#### 8. MODEL TESTING & EVALUATION:
> - Used the Dummy Classifier to calculate the Model Accuracy and Racall: Train and Test Accuracy is 14%
> - Build up the Simple Model: Logistics Regression, KNN, Decision Trees, Random Forest, XGBOOST and Artificial Neural Network(ANN): Random Forest and XGBOOST have the best Test accuracy of 89% and 88% respectively

| Simple Models | Train Accuracy | Test Accuracy | Test Precision | Test Recall |
|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.56 | 0.54 | 0.53 | 0.54 |
| K-nearest Neighbor | 0.88 | 0.81 | 0.80 | 0.81 |
| Decision Tree | 1.00 | 0.82 | 0.82 | 0.82 |
| Random Forest | 1.00 | 0.89 | 0.89 | 0.89 |
| XGBoost | 0.99 | 0.88 | 0.88 | 0.88 |
| Neural Network | 0.64 | 0.63 | 0.62 | 0.64 |

> - Also tested the **SVM model separately and the Train and Test accuracy was 84% and 79% respectively**. Not included in the Jupyter notebook because of the runtime crossing 10 minutes and size of the notebook exceeded 4MB.
> - The accuracies for Linear/Distance based models(KNN, Logistics Refression) is low as they account for Linear features, the hyperparameter tuning is done for only Decision Trees, random Forest, XGBOOST and ANN.
> - Did the Hyper Parameter tuning for Decision Trees, Random Forest, XGBOOST and ANN models: **Test Accuracy and Recall for Random Forest and XGBOOST in the range of 89%-90%**. **Random FOrest seems to the best Model with Highest accuracy of 90% and Recall of 89%**
> - Selected the **RANDOM FOREST** model as it had **best accuracy(90%) and Recall(90%)**, but more importantly, its easy to communicate the non-technical audience on how the model works and what really drives the Quality of Wine.
> - Used the **Confusion Matrix** to show the **Accuracy, Recall and Precision**  
#### 9. RESULTS / FINDINGS
| Model | Train Accuracy | Test Accuracy | Test Precision | Test Recall |
|:---:|:---:|:---:|:---:|:---:|
| Best Decision Tree | 1.00 | 0.82 | 0.81 | 0.81 |
| Best Random Forest | 1.00 | 0.90 | 0.89 | 0.90 |
| Best XGBoost | 1.00 | 0.90 | 0.90 | 0.90 |
| Best Neural Network | 0.74 | 0.70 | 0.69 | 0.69 |
> - Best Model: **RANDOM FOREST**: Balance between Recall, Precision and Accuracy (All 90%)
>>> - Training Accuracy: 1.00
>>> -	**Testing Accuracy: 0.90**
>>> - **Testing Recall: 0.90**
>>> - **Testing Recall for Class 5: 0.80**
>>> - **Testing Recall for Class 6: 0.65**
>>> - Testing Precision: 0.90
> - Best Parameter of Random Forest Model:{'classifier__max_depth': None, 'classifier__max_features': 'sqrt', 'classifier__min_samples_leaf': 1, 'classifier__min_samples_split': 2, 'classifier__n_estimators': 200}
> - **Random Forest Model perform better for White Wine as compared to Red Wine** with accuracy of 90% vs 84%
>>> - Recall for Class 5 for Red wine (88%) is much higher as compared to White Wine (76%)
>>> - Recall for Class 6 for White Wine (66%) is much higher as compared to Red Wine (57%)
> - **"Color of Wine" seems to have a very little impact on the Wine Quality**. The Random Forest model accuracy reduces by just 1% (from 90% to 89%) if the "Color" feature is not included in the Model building and finetuning.
> - Factors that influence the Red and White Wine Quality then most are (Excluding "Color"):
>>> - "Chlorides"
>>> - "Free Sulfur Dioxide"
>>> - "Alcohol"
>>> - "Volatile Acidity"
>>> - "pH"
>>> - "Fixed Acidity"
>>> - "Citric Acid"
>>> - "Residual Sugar"
>>> - "Sulphates"
#### 10. NEXT STEPS
> - Identify other external factors that can influence the quality of wine like weather, soil, Flavanoids, Magnesium content etc and include them in the model that may help to improve the Recall for Class 5 and 6 
> - Identify Potential use cases where wine classification model can be applied in practice.
> - Understand the business objectives and align the model accordingly basis the use case.

#### 11. PRACTICAL APPLICATION AND BUSINESS VALUE OF WINE QUALITY PREDICTION MODEL
> - classify wine into different categories (1 through 10) and at scale without relying on manual process of classifying wine that’s time consuming and is left to individual choices and taste buds at times.
> - Opportunity for Wine makersto modify and finetune the production process so that the Wine falls into desired class (4,5,6 into medium category, 7,8 into high and 9 and 10 into fine wine category)
> - From a marketeer perspective, it provides opportunity to maximize the revenue or avoid revenue leakage for wines misclassified into lower grade than it actually is.


#### -------------------------------------------END OF REPORT------------------------------------------- ###
