# Logistic Classifier for Predicting Customer Churn in SyriaTel

# Overview
The telecommunications industry, specifically SyriaTel, faces challenges with customer retention leading to financial losses. This project aims to utilize data science techniques to build a predictive model that can identify potential churners - customers who are likely to discontinue their services with SyriaTel. By understanding patterns in customer behavior, this model intends to assist the telecom business in minimizing losses associated with customer attrition.

## Business Understanding
Telecom companies rely on retaining customers for sustainable growth. Understanding customer behavior and predicting churn can aid in developing strategies to retain existing customers. This initiative seeks to provide actionable insights to stakeholders, including SyriaTel and other telecom entities, to refine customer engagement strategies, reduce churn, and enhance customer loyalty.

## Problem Statement
The primary challenge is to develop a classification model that accurately predicts whether a customer is likely to churn or not. Predicting churn patterns will enable SyriaTel to take proactive measures, such as personalized retention strategies or targeted promotions, to mitigate customer attrition.

## Objectives
- **Predictive Model**: Develop a classifier that predicts churn with at least 85% accuracy, assisting SyriaTel in identifying customers at risk of leaving.
- **Identifying Key Features**: Determine the main factors influencing churn rate to guide targeted retention efforts. Two prominent features contributing to churn prediction will be identified.

## Description of Dataset
The dataset sourced from Kaggle contains customer-related information, such as usage patterns, account details, and services subscribed, which will be leveraged for model development. It consists of several features and a binary target variable indicating whether a customer churned or not.
- **state**: This column likely represents the state of the customer's residence or location. It contains categorical data indicating the state where the customer resides.

- **account length**: This numerical column indicates the duration or length of time the customer has been associated with SyriaTel. It represents the number of days the customer has had an active account.

- **area code**: This categorical column denotes the area code associated with the customer's phone number. It may serve as a regional identifier for the customer's location.

- **phone number**: This column contains unique identifiers (strings) for each customer's phone number. It's likely used for identification purposes and may not directly contribute to churn prediction.

- **international plan**: A categorical column indicating whether the customer has subscribed to an international calling plan. It's likely a binary (yes/no or True/False) variable.

- **voice mail plan**: Similar to the international plan, this column likely indicates whether the customer has a subscription to a voicemail plan. It's also a binary categorical variable.

- **number vmail messages**: This numerical column represents the count of voicemail messages received by the customer.

- **total day minutes**: Numeric data representing the total number of minutes the customer used during daytime calls.

- **total day calls**: Numeric data representing the total number of calls made by the customer during the day.

- **total day charge**: Numeric data indicating the total charges incurred by the customer for daytime calls.

- **total eve minutes**: Similar to 'total day minutes', this column represents the total number of minutes used during evening calls.

- **total eve calls**: Total number of calls made by the customer during the evening.

- **total eve charge**: Total charges incurred by the customer for evening calls.

- **total night minutes**: Total minutes used during nighttime calls.

- **total night calls**: Total number of calls made during the night.

- **total night charge**: Total charges for nighttime calls.

- **total intl minutes**: Total minutes used for international calls.

- **total intl calls**: Total number of international calls made.

- **total intl charge**: Total charges for international calls.

- **customer service calls**: Number of calls made to customer service by the customer.

- **churn**: The target variable indicating whether the customer churned (discontinued services) or not. It's a binary variable (likely 'yes' or 'no').


## 3. Data Preprocessing
- **Data Cleaning**: No missing or duplicate values.
- **Feature Engineering**:
  - Extracted phone area code from 'phone number'.
  - Set 'phone number' as index.
  - Created new features: 
    - Average call duration for different call times.
    - Total charges per call.
    - Customer interaction index.
    - Call charge to minute ratio.
    - Total Activity Index.

## 4. Exploratory Data Analysis (EDA)
- Utilized custom distribution and count plots functions to visualize data distributions.
  - Observed that continuous variables mostly followed a normal distribution, showcasing Gaussian curves.
- Identified insights regarding the distribution patterns of different features and their potential impact on the project's objectives.

## Feature Engineering 2
- Expanded feature set by creating new columns:
  - **Average Call Duration for Different Call Times**:
    
    df['avg_day_call_duration'] = df['total day minutes'] / df['total day calls']
    df['avg_eve_call_duration'] = df['total eve minutes'] / df['total eve calls']
    df['avg_night_call_duration'] = df['total night minutes'] / df['total night calls']
    df['avg_intl_call_duration'] = df['total intl minutes'] / df['total intl calls']
    ```
  - **Total Charges per Call**:
    
    df['day_charge_per_call'] = df['total day charge'] / df['total day calls']
    df['eve_charge_per_call'] = df['total eve charge'] / df['total eve calls']
    df['night_charge_per_call'] = df['total night charge'] / df['total night calls']
    df['intl_charge_per_call'] = df['total intl charge'] / df['total intl calls']
    ```
  - **Customer Interaction Index**:
   
    cols_to_sum = ['customer service calls', 'number vmail messages', 'total day calls', 'total eve calls', 'total night calls', 'total intl calls']
    df['interaction_index'] = df[cols_to_sum].sum(axis=1)
    ```
  - **Call Charge to Minute Ratio**:
    
    df['day_charge_minute_ratio'] = df['total day charge'] / df['total day minutes']
    df['eve_charge_minute_ratio'] = df['total eve charge'] / df['total eve minutes']
    df['night_charge_minute_ratio'] = df['total night charge'] / df['total night minutes']
    df['intl_charge_minute_ratio'] = df['total intl charge'] / df['total intl minutes']
    ```
  - **Total Activity Index**:
    df['total_activity_index'] = df[['total day minutes', 'total eve minutes', 'total night minutes', 'total intl minutes']].sum(axis=1)
    ```
- Addressed missing values generated by these operations.

## Data Splitting
- Split the dataset into 'train_df' and 'test_df' using sklearn for modeling purposes.

## Correlation Analysis
- Employed the chi-squared test for categorical features.
- Conducted point-biserial correlation for continuous values.
- Eliminated features with p-values > 0.05.

## Collinearity Analysis
- Developed a function to handle collinear features.
- Obtained a 'clean_df' with shape (2499, 10) after dropping certain features due to correlation and collinearity.

## Class Imbalance Handling
- Employed SMOTE technique to address class imbalance in the dataset.

## Dimensionality Reduction (PCA)
- Utilized PCA (Principal Component Analysis) for dimensionality reduction.
- Determined the number of components for PCA to be 9.
## Modeling
- Ensured consistency in column numbers between 'train_df' and 'test_df'.
- Utilized various machine learning models:
  - **Logistic Regression**
  - **DecisionTreeClassifier**
  - **KNNs**
  - **Random Forests**
  - **XG-Boost Classifier**
  - **Ensembling**
- Evaluated model performances based on:
  - Accuracy
  - Precision
  - Recall
  - F1-score

## Model Performance
- **XGBoost**: Achieved an accuracy of 92.45% with notable precision and recall scores, indicating good overall performance.
- **Random Forest**: Slightly lower accuracy at 91.61% but balanced precision and recall scores.
- **KNN**: Lower accuracy at 82.37% with lower precision and moderate recall.
- **Decision Tree**: Performed less accurately at 76.74% with lower precision and recall.
- **Meta_learner_scaled_scores**: Performed the best with an accuracy of 93.17%, high precision, recall, and F1 score.

## Top Predictors of Churn Rate
- **International Plan & Voice Mail Plan**: Customers subscribed to an international plan or a voicemail plan may exhibit different usage patterns or behavior leading to a higher likelihood of churn. For instance, dissatisfaction with international call quality or not utilizing voicemail services could influence churn.
- **Total Day Charge, Total Intl Calls, Total Intl Charge**: Higher charges incurred during the day or for international calls might lead to dissatisfaction, prompting customers to consider switching service providers.
- **Customer Service Calls**: Customers frequently reaching out to customer service might indicate issues or dissatisfaction with the service, which could correlate with a higher churn rate.
- **Day Charge-Minute Ratio & Intl Charge-Minute Ratio**: Higher ratios of charges to minutes might signify expensive plans relative to usage, potentially causing dissatisfaction and prompting customers to churn.
- **State Target Encoded**: Encoded state information could capture regional tendencies or variations in customer behavior that influence churn rate, like local service quality or competitive offerings in different areas.

## Conclusion & Recommendations

1. **Focus on Customer Experience:**
   - **Explanation:** Address customer concerns related to international calling services and customer service interactions. International plan subscription and customer service calls are key predictors of churn. Improving these areas directly impacts customer satisfaction, leading to a reduction in churn rates.

2. **Offer Customized Plans:**
   - **Explanation:** Evaluate pricing structures for day and international calls based on customer usage patterns. High charges for day and international calls are significant predictors of dissatisfaction. Aligning pricing to match actual usage prevents customer dissatisfaction, effectively reducing the probability of churn.

3. **Enhance Service Quality:**
   - **Explanation:** Invest in improving service quality across different states, considering regional disparities in churn rates. State-target encoded data suggests varying service satisfaction levels. Focusing on enhancing service quality in regions with higher churn rates directly addresses predictors associated with dissatisfaction and potential customer loss.

4. **Promote Engagement:**
   - **Explanation:** Encourage adoption of value-added services, specifically voicemail plans, known to contribute positively to customer engagement. Increasing engagement aligns with predictors indicating lower churn rates. Promotional efforts targeted at such services enhance customer satisfaction, potentially reducing churn.

5. **Continuous Monitoring:**

   - **Explanation:** Regularly monitor customer feedback and usage patterns to adapt strategies and services. Continuously evolving strategies based on evolving customer needs aligns with predictors of higher customer satisfaction and lower churn rates. Continuous monitoring ensures prompt adaptation to changing customer preferences, ultimately reducing churn.


## Model Deployment
- Saved the trained model using joblib.
- Created a 'streamlit.py' file to load the 'model.joblib' file and run it with Streamlit.
- To use the app on GitHub:
  1. Clone the repository.
  2. Ensure required Python libraries are installed by running:
     ```
     pip install -r requirements.txt
     ```
  3. Run the Streamlit app using the following command:
     ```
     streamlit run streamlit.py
     ```
  4. Access the app via the provided URL in the terminal.

The 'streamlit.py' file loads the pre-trained model and enables users to interact with the model for predictions or analysis.
