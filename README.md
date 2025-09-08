### CapstoneProject : Wine Classification (Quality Ratings - 1 through 10)
#### Link to Jupyter Notebook: https://github.com/deepakkaushik72/Capstone_Wine_Quality_Classification/blob/main/red_white_classification.ipynb
#### Link to Images: https://github.com/deepakkaushik72/Capstone_Wine_Quality_Classification/tree/main/Images
#### Link to Data: https://github.com/deepakkaushik72/Capstone_Wine_Quality_Classification/blob/main/winequality-red.csv
####               https://github.com/deepakkaushik72/Capstone_Wine_Quality_Classification/blob/main/winequality-white.csv

1. BUSINESS PROBLEM / OBJECTIVE:

Help the bank to target potential customers more effectively and improve their customer target effectiveness (Conversion Rate) and optimize the Return on Marketing Investment (ROMI). Have a Targeted marketing campaign than mass targeting. Precise trade off decision on "Revenue Maximization and Conversion Rate (Precision- Recall Tradeoff)"
The business goal is to find a model that can predict in advance whether a customer will subscribe to a term deposit based on their demographic, history and social and economic attributes (21 Numerical and Categorical variables in Dataset)
Better Management of available resources (human effort, phone calls and time)
Over 80% Accuracy and with 70% Recall of the positive class (converted customers to subscription)
2. APPROACH

This is a binary classification problem where the target variable is 'y' which indicates whether the customer has subscribed to a term deposit or not (1 - Yes, 0 - No).
Will be using the Supervised Machine Learning Classification models.
The model will be trained on a dataset containing various features such as age, job, marital status, education, default status, housing loan status, personal loan status, contact communication type, last contact month of the year, number of contacts performed during this campaign, number of days since the customer was last contacted from a previous campaign, number of contacts performed before this campaign and the outcome of the previous marketing campaign.
Different classification models like Logistics Regression, Decision Tree, KNN and SVM will be used to predict the target variable
3. SUCCESS CRITERION:

Deploy the model will predict whether a customer will Subscribe to Term Deposit or not with the following success criteria:

Clearly identfy the factors/ features that drive the conversion rate
Highest Recall (70%) of the positive class (Y=1) while keeping accuracy over 80%
Maximizing the conversion rate (Precision) of the Marketing campaign and optimizing the Return on Marketing Investment
Optimizing the F1 Score (Precision-Recall Tradeoff)
4. DATA PREPARATION:

There are 41188 records in the file: “bank-additional-full.csv”
There are 12 duplicate records that have to be deleted
Imbalanced class distribution as there are only 12% in the positive class (only 12% of the total customers reached subscribed to the Term Deposit)
Looked at 21 features (10 numerical and 11 categorical including Target variable)
Job, Marital, education, default, housing, loan have “Unknown Values”, so these are missing values and hence dropped from the Dataset. Dropped 10,400+ records from the dataset. Final dataset/Data frame called “Bank” has only 30478 records
Dropped the “Duration” column out from the final data frame as We will never know the duration before it happens
5. DATA UNDERSTANDING:

Most important Numerical features are: 'previous', 'pdays', 'emp.var.rate', 'euribor3m', 'nr.employed' and are negatively correlated to the conversion rate
Age, Campaign and Consumer confidence Index show extremely low correlation with the conversion rate.
Most important Categorical features are: Job', 'Education', 'Default', 'Month', 'Poutcome'
"No Default" have a very significant impact on Conversion Rate
"Student" and "Retired" have higher conversion rates (over 25%)
"Illiterate" category have a higher conversion rate (over 25%) compared to other education levels
"Sept, Oct, Dec and March" much higher conversion rate (40% - 55%)
"Poutcome": Success of the previous campaign outcome leads to higher conversion rates (over 60%)
6. FEATURE ENGINEERING:

There are 10 Features Identified (incl. target Variable) for building the ML models: These are nr.employed, emp.var.rate, pdays, previous, default, job, education, poutcome, month and Target variable (y)
Defined and Input Features and Target Variable
Created a preprocess for Scaling the Numerical features and OneHotEncoding for categorical features
Created a Train and Test Split of the Input features and Target Variable (Applied 25% test split and Stratify = y)
7. MODELING TESTING & EVALUATION:

Used the Dummy Classifier to calculate the Model Accuracy and Racall: Accuracy at 87% Recall is 0%
Build up the Simple Model: KNN, Decision Trees, Logistic Regression and SVM: All have a base accuracy of around 87% and Recall is just 20%
Did the Hyper Parameter tuning for all the models: Accuracy in the range of 81%-82% and Recall improved significantly to 63% for all models except KNN
Selected the DECISION TREE model as it had an accuracy of 81%, Recall of 63% and Precision of 38% but more importantly, its easy to communicate the non-technical audience on how the model works to identify whether a customer will subscribe to Term Deposit or Not (Target variable 1 or 0)
Used the Confusion Matrix to show the Recall for both Positive and Negative class using all the Classification Models
8. RESULTS / FINDINGS

Best Model: Decision Tree: Balance between Recall, Precision and Accuracy
Best Parameters: {'classifier__max_depth': 5, 'classifier__min_samples_split': 10}
Training Accuracy: 0.85
Testing Accuracy: 0.82
Testing Recall: 0.63
Testing Precision: 0.38
Factors that influence the Decision to Subscribe to Term Deposit:"nr.employed", "pdays", "previous", "month", "poutcome"
Student and Retired categories have twice the conversion rate (~25%+)
"Illiterate" category in Education has twice the conversion rate (~25%+)
"September", "October", "December" and "March" have conversio rate of over 45%
"Outcome of Previous campaign (poutcome)" have significantly higher conversion rate of over 60%
9. DEPLOYMENT / NEXT STEPS

Communicate the Results of the best Model
Understand the Business context in terms of what other factors may influence the Conversion rate and get the additional data and model those features into the Data set. Some of the other Features than can be considered are: "Account Balance", competition data around "Interest Rates for Deposits", "Debit card", "Agent name who made the call", "Awareness of the Bank product", "Phone Conversation transcript"
Aim for 90% Accuracy and 80% Recall with new features included in the dataset
Deploy the current model (Decision Tree) working closely with The Marketing team understanding the Business objectives they would want to achieve.
Work with Marketing Team to share the Results and understand the Marketing Objectives:
Maximize Revenue (Recall will be high but conversion rate at current levels of 12.6% reflecting through precision): sub optimal scenarios resulting in wastage of resources and Marketing dollars as a result of Mass Campaign
Maximize the Marketing Investment with Targeted campaign (Coverage of 63% customers (Recall) and with a conversion rate of 38% (same as precision) which is 3 times the current)
Maximize the revenue with double the conversion Rate: Lower the threshold to 0.4, which will cover 70% customers converted (Recall of 70%) and have a conversion rate of 25% (Twice the current conversion rate of 12.6%)
