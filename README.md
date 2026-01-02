## Ola Driver Churn Prediction:

### Problem Statement
Recruiting and retaining drivers is a significant challenge for Ola, with high churn rates impacting organizational morale and leading to costly driver acquisition processes. Drivers frequently switch services or stop working, making it crucial to predict attrition.

### Business Objective
To predict whether a driver will leave the company based on their demographic information, tenure data, and historical performance (e.g., quarterly rating, business acquired, grade, income). This prediction will help Ola implement targeted retention strategies and reduce the financial burden of new driver acquisition.

### Column Profiling

*   **MMM-YY**: Reporting Date (Monthly)
*   **Driver_ID**: Unique ID for drivers
*   **Age**: Age of the driver
*   **Gender**: Gender of the driver – Male: 0, Female: 1
*   **City**: City Code of the driver
*   **Education_Level**: Education level – 0 for 10+, 1 for 12+, 2 for graduate
*   **Income**: Monthly average Income of the driver
*   **Date Of Joining**: Joining date for the driver
*   **LastWorkingDate**: Last date of working for the driver
*   **Joining Designation**: Designation of the driver at the time of joining
*   **Grade**: Grade of the driver at the time of reporting
*   **Total Business Value**: The total business value acquired by the driver in a month (negative business indicates cancellation/refund or car EMI adjustments)
*   **Quarterly Rating**: Quarterly rating of the driver: 1, 2, 3, 4, 5 (higher is better)

### Data Preprocessing

**1. Null Value Treatment:**
*   `LastWorkingDate`: Filled with `0` (indicating the driver is still working) if NaN, otherwise `1` (indicating they have left).
*   `Age` and `Gender`: Missing values were filled using the `ffill` method to propagate the last valid observation forward.

**2. Feature Engineering:**
*   **Date Column Splitting**: The `MMM-YY` column was split into `r_month`, `r_day`, and `r_year`. Similarly, `Dateofjoining` was split into `j_day`, `j_month`, and `j_year`.
*   **City Code Extraction**: The `City` column (e.g., 'C23') was converted to a numerical representation by extracting the integer part (e.g., '23').
*   **Total Experience in Days**: A new feature `TotalexpinDays` was created by calculating the difference in days between `MMM-YY` and `Dateofjoining`.
*   **Aggregation and New Features (after grouping by `Driver_ID`)**:
    *   `TotalexpMonths`: Count of `MMM-YY` for each driver, representing total months served.
    *   `tot_income`: Sum of `Income` for each driver.
    *   `avg_income`: `tot_income` divided by `TotalexpMonths`.
    *   `hasNegBusiValue`: A binary indicator (`1` if a driver had any negative `Total Business Value`, `0` otherwise) to capture potential issues like cancellations/refunds.
    *   `totBusiValue`: Sum of `Total Business Value` for each driver.
    *   `Quarterly Rating`: Sum of `Quarterly Rating` (used as a proxy for total rating over tenure).
    *   `Grade`: Maximum `Grade` achieved by the driver.
    *   `City`: Mean `City` value (after numerical conversion).
    *   `Age`: Maximum `Age` recorded for the driver.

**3. Encoding:**
*   Categorical features such as `Gender`, `Joining Designation`, `TotalexpMonths`, `Age`, `City`, `Education_Level`, `Grade`, `Quarterly Rating`, and `hasNegBusiValue` were transformed using `LabelEncoder`. This was deemed sufficient as most were already ordinal or had a limited number of unique integer values.

### Exploratory Data Analysis (EDA)

**1. Univariate Analysis:**
*   **Target Variable (`LastWorkingDate`)**: The dataset shows a significant imbalance, with approximately 91.5% of records indicating drivers are `Not leaving` (value 0) and 8.5% indicating `Leaving` (value 1).
*   **Numerical Features (`Income`, `Total Business Value`)**: Box plots revealed the presence of outliers in both `Income` and `Total Business Value`.
*   **Categorical Features (`Age`, `Gender`, `City`, `Education_Level`, `Joining Designation`, `Grade`, `Quarterly Rating`)**: Bar plots showed varying distributions:
    *   `Gender`, `City`, `Education_Level`: These variables showed a relatively even distribution in relation to `LastWorkingDate`.
    *   `Joining Designation`: Driver churn was more pronounced for designations 2 and 3.
    *   `Grade`: Churn was predominantly observed in Grades 1 and 2, with much lower rates for Grades 3, 4, and 5.
    *   `Quarterly Rating`: A quarterly rating of 1 showed a higher dependency on `LastWorkingDate` (i.e., higher churn), while other ratings had less impact.

**2. Bivariate Analysis:**
*   When comparing categorical features against `LastWorkingDate`, the bar plots for `LastWorkingDate` (value 0 vs. 1) reinforced the class imbalance, making it difficult to draw definitive conclusions directly from these plots without addressing the imbalance. However, general trends (as noted in univariate analysis) still held.

**3. Correlation Analysis:**
*   The heatmap indicated that the target variable `LastWorkingDate` did not show a strong linear correlation with many of the other columns. This suggests that the relationship might be complex and non-linear, making it challenging for simple linear models but potentially more suitable for tree-based or ensemble methods.

### Outlier Treatment

Outliers were observed in both `Income` and `Total Business Value` through box plots. However, these outliers were not explicitly treated (e.g., capping or removal) for the following reasons:
*   Capping extreme values would compress the data points significantly, particularly for `Total Business Value` where the mean is close to zero, potentially distorting the true distribution and relationships within the data.
*   Dropping rows with outliers would lead to a substantial loss of data, which is undesirable given the existing class imbalance and the nature of the problem.

Therefore, it was decided to proceed without aggressive outlier treatment, allowing the models (especially tree-based ensembles) to handle these variations naturally.

### Model Training and Evaluation

The models were trained and evaluated on both imbalanced and SMOTE-balanced datasets to assess the impact of class balancing on performance.

**1. Logistic Regression**
*   **Without SMOTE (Imbalanced Data)**:
    *   Training Score: 0.751
    *   Test Score: 0.769
    *   Classification Report:
        *   Class 0 (Not Leaving): Precision = 0.78, Recall = 0.40, F1-score = 0.53
        *   Class 1 (Leaving): Precision = 0.77, Recall = 0.95, F1-score = 0.85
    *   *Observation*: Poor recall for class 0, indicating many drivers who left were misclassified as staying.
*   **With SMOTE (Balanced Data)**:
    *   Training Score: 0.687
    *   Test Score: 0.771
    *   Classification Report:
        *   Class 0 (Not Leaving): Precision = 0.66, Recall = 0.59, F1-score = 0.63
        *   Class 1 (Leaving): Precision = 0.81, Recall = 0.86, F1-score = 0.84
    *   *Observation*: Improved recall for class 0, but a slight drop in precision and overall scores compared to class 1.

**2. K-Nearest Neighbors (KNN)**
*   **Without SMOTE (Imbalanced Data)**:
    *   Training Score: 0.803
    *   Test Score: 0.736
    *   Classification Report:
        *   Class 0: Precision = 0.62, Recall = 0.45, F1-score = 0.53
        *   Class 1: Precision = 0.77, Recall = 0.87, F1-score = 0.82
    *   *Observation*: Similar to Logistic Regression, poor performance for class 0.
*   **With SMOTE (Balanced Data)**:
    *   Training Score: 0.827
    *   Test Score: 0.671
    *   Classification Report:
        *   Class 0: Precision = 0.49, Recall = 0.61, F1-score = 0.54
        *   Class 1: Precision = 0.79, Recall = 0.70, F1-score = 0.74
    *   *Observation*: Performance significantly worsened after balancing, particularly for precision of class 0.

**3. Decision Tree Classifier**
*   **Without SMOTE (Imbalanced Data)**:
    *   Training Score: 1.0
    *   Test Score: 0.769
    *   Classification Report:
        *   Class 0: Precision = 0.64, Recall = 0.65, F1-score = 0.65
        *   Class 1: Precision = 0.83, Recall = 0.83, F1-score = 0.83
    *   *Observation*: Severe overfitting, but reasonable F1-scores on test data.
*   **With SMOTE (Balanced Data)**:
    *   Training Score: 1.0
    *   Test Score: 0.769
    *   Classification Report:
        *   Class 0: Precision = 0.62, Recall = 0.72, F1-score = 0.67
        *   Class 1: Precision = 0.86, Recall = 0.79, F1-score = 0.82
    *   *Observation*: Still highly overfit, slight improvement in class 0 recall.

**4. Random Forest Classifier**
*   **Without SMOTE (Imbalanced Data)**:
    *   Training Score: 1.0
    *   Test Score: 0.826
    *   Classification Report:
        *   Class 0: Precision = 0.81, Recall = 0.60, F1-score = 0.69
        *   Class 1: Precision = 0.83, Recall = 0.93, F1-score = 0.88
    *   *Observation*: High overfitting with good overall accuracy. F1-score for class 0 is lower.
*   **With SMOTE (Balanced Data)**:
    *   Training Score: 1.0
    *   Test Score: 0.813
    *   Classification Report:
        *   Class 0: Precision = 0.71, Recall = 0.71, F1-score = 0.71
        *   Class 1: Precision = 0.86, Recall = 0.86, F1-score = 0.86
    *   *Observation*: Better balance in F1-scores between classes compared to imbalanced data, but still overfitting.

**5. Bagging Classifier**
*   **Without SMOTE (Imbalanced Data)**:
    *   Training Score: 0.986
    *   Test Score: 0.811
    *   Classification Report:
        *   Class 0: Precision = 0.71, Recall = 0.71, F1-score = 0.71
        *   Class 1: Precision = 0.86, Recall = 0.86, F1-score = 0.86
    *   *Observation*: Relatively good F1-scores for both classes, but training score indicates some overfitting.

**6. XGBoost Classifier**
*   **Without SMOTE (Imbalanced Data)**:
    *   Training Score: 0.999
    *   Test Score: 0.822
    *   Classification Report:
        *   Class 0: Precision = 0.76, Recall = 0.66, F1-score = 0.71
        *   Class 1: Precision = 0.85, Recall = 0.90, F1-score = 0.87
    *   *Observation*: Strong performance, but high training score suggests potential overfitting.
*   **Hyperparameter Tuned XGBoost (on Imbalanced Data)**:
    *   Best Parameters: `{'subsample': 0.8, 'n_estimators': 200, 'max_depth': 2, 'learning_rate': 0.07, 'colsample_bytree': 0.6}`
    *   Training Score: 0.861
    *   Test Score: 0.830
    *   Classification Report:
        *   Class 0: Precision = 0.81, Recall = 0.62, F1-score = 0.70
        *   Class 1: Precision = 0.84, Recall = 0.93, F1-score = 0.88
    *   *Observation*: Significantly reduced overfitting with improved and balanced test scores. ROC-AUC score of 0.902.

**7. LightGBM Classifier**
*   **Without SMOTE (Imbalanced Data)**:
    *   Training Score: 0.989
    *   Test Score: 0.836
    *   Classification Report:
        *   Class 0: Precision = 0.80, Recall = 0.66, F1-score = 0.72
        *   Class 1: Precision = 0.85, Recall = 0.92, F1-score = 0.88
    *   *Observation*: Good performance with high test accuracy, but high training score.

### Analysis and Recommendation:

#### 1. Initial Models and Issues
   - **Logistic Regression**:
     - Precision and recall show imbalance in class performance, with better performance for class 1 (drivers who leave). However, precision and recall for class 0 (drivers who stay) are notably low.
     - Even after balancing with **SMOTE**, precision and recall for class 0 improved but are still suboptimal.
   - **KNN**:
     - The performance after balancing data showed a significant decrease in accuracy compared to the unbalanced dataset. Precision and recall for class 0 were quite low.
   - **Decision Tree**:
     - The model displayed overfitting (training score 1.0), though recall and precision were relatively balanced.
   - **Random Forest**:
     - Before balancing, Random Forest showed good accuracy but indicated overfitting (training score of 1.0).
     - After balancing and hyperparameter tuning, the model continued to show a decent performance, with improvements in precision and recall for both classes.
   - **Bagging**:
     - The model's accuracy is good but slightly lower than Random Forest, with class 0 suffering from lower precision and recall compared to class 1.
   - **XGBoost**:
     - After tuning, XGBoost gave the best balance of precision and recall, especially for class 1. However, precision for class 0 could still be improved.
   - **LGBMClassifier**:
     - Similar performance to XGBoost, showing good accuracy and precision, but class 0 still suffers slightly in recall.

#### 2. Recommendations:

   - **Best Model**: Based on the overall performance metrics (precision, recall, F1-score, and accuracy), **XGBoost** is the best model for predicting driver attrition. It shows a strong balance between both classes, with good recall and precision.
   
   - **Model Performance**:
     - **XGBoost** handled the imbalanced data well, with fewer false positives and false negatives than the other models. After tuning, it provided the best results with an accuracy of 83%.
     - **Random Forest** also performed well but showed overfitting tendencies before hyperparameter tuning.

   - **Handling Imbalanced Data**:
     - The use of **SMOTE** was beneficial in addressing class imbalance, improving recall and precision for the minority class. However, balancing techniques should be combined with **precision-recall trade-off analysis** to avoid overfitting.
     - **XGBoost** and **LGBMClassifier** are naturally robust to imbalanced datasets and handle it better, as demonstrated by their results.

#### 3. Future Improvements:
   - **Feature Engineering**: Focus on deriving new features like interaction terms, city-specific trends, or income volatility to capture more driver behavior patterns.
   - **Time-Series Modeling**: Explore using time-based features such as how a driver's performance or income changes over time, which could be beneficial for models like **LSTM** or **RNN**.
   - **Ensemble Approaches**: Combining **XGBoost** with **Random Forest** or using voting classifiers might yield better performance.

#### 4. Business Recommendations:
   - **Driver Retention Strategies**: Focus on drivers who are at risk of leaving based on model predictions. Interventions such as bonuses or incentives can be targeted to drivers flagged by the model.
   - **Dynamic Incentive Structures**: Adjusting rates dynamically based on the driver’s performance, tenure, or churn risk can improve retention.
   - **New Driver Acquisition**: While new driver acquisition is costly, focus on retaining current drivers through **personalized retention programs** informed by the model's outputs.

#### **Conclusion:**
   - **XGBoost** is the optimal model for predicting driver attrition due to its balanced performance and robust handling of imbalanced data. The focus should now shift toward refining feature engineering and retention strategies based on model predictions.
