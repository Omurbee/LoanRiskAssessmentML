# ALL CODES COMMENTED FOR EASE

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

st.title("Loan Risk Prediction Web App")

# Function to map LoanStatus to RiskCategory
def map_loan_status(status):
    if status in ['Current', 'Completed']:
        return 'Low Risk'
    elif status in ['FinalPaymentInProgress', 'Past Due (1-15 days)', 'Past Due (16-30 days)', 'Past Due (31-60 days)', 'Past Due (61-90 days)', 'Past Due (91-120 days)']:
        return 'Medium Risk'
    elif status in ['Chargedoff', 'Defaulted', 'Past Due (>120 days)', 'Cancelled']:
        return 'High Risk'

# Function to map RiskCategory to the corresponding PRS column
def get_prs(risk_category, high_risk_prs, low_risk_prs, medium_risk_prs):
    if risk_category == 'High Risk':
        return high_risk_prs
    elif risk_category == 'Low Risk':
        return low_risk_prs
    elif risk_category == 'Medium Risk':
        return medium_risk_prs
    else:
        return np.nan  # Return NaN if the risk category is not recognized

# Upload CSV file
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file).head(2000)

    # Filtered columns after PCA multicollinearity test
    filtered_columns = [
        'Term', 'LoanStatus', 'BorrowerAPR', 'EstimatedReturn',
        'ProsperRating (numeric)', 'ListingCategory (numeric)',
        'BorrowerState', 'Occupation', 'EmploymentStatus',
        'IsBorrowerHomeowner', 'CreditScoreRangeLower',
        'OpenCreditLines', 'TotalCreditLinespast7years',
        'OpenRevolvingAccounts', 'OpenRevolvingMonthlyPayment',
        'InquiriesLast6Months', 'CurrentDelinquencies',
        'AmountDelinquent', 'PublicRecordsLast10Years',
        'RevolvingCreditBalance', 'BankcardUtilization',
        'DebtToIncomeRatio', 'IncomeRange', 'IncomeVerifiable',
        'StatedMonthlyIncome', 'LoanOriginalAmount',
        'MonthlyLoanPayment'
    ]
    sliced_data = df[filtered_columns]

    # Handling missing and negative values
    numerical_cols = sliced_data.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_cols:
        sliced_data[col].fillna(sliced_data[col].mean(), inplace=True)

    categorical_cols = sliced_data.select_dtypes(include=['object', 'bool']).columns
    for col in categorical_cols:
        sliced_data[col].fillna(sliced_data[col].mode()[0], inplace=True)

    # Convert negative values to zero
    sliced_data[numerical_cols] = sliced_data[numerical_cols].applymap(lambda x: max(x, 0))

    # Map LoanStatus to RiskCategory
    sliced_data['RiskCategory'] = sliced_data['LoanStatus'].apply(map_loan_status)

    # Encode categorical variables using LabelEncoder
    le = {}
    for col in categorical_cols:
        le[col] = LabelEncoder()
        sliced_data[col] = le[col].fit_transform(sliced_data[col])

    # Prepare the features (X) and target (y)
    X = sliced_data.drop('RiskCategory', axis=1)
    y = sliced_data['RiskCategory']

    # Encode target variable
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)

    # Initialize the RandomForest model with a pipeline that includes ADASYN
    adasyn = ADASYN(random_state=42)
    rf_model = ImbPipeline([
        ('scaler', StandardScaler()),
        ('adasyn', adasyn),
        ('classifier', RandomForestClassifier(
            random_state=42,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced'
        ))
    ])

    # Defining the KFold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Initializing an array to store the risk scores
    personalized_risk_scores = np.zeros((len(sliced_data), 3))  # Assuming 3 classes

    # Perform KFold Cross-Validation
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]

        rf_model.fit(X_train, y_train)
        y_prob = rf_model.predict_proba(X_test)
        personalized_risk_scores[test_index] = y_prob

    # Appending the personalized risk scores to the dataframe
    sliced_data['PRS_HighRisk'] = personalized_risk_scores[:, 0]
    sliced_data['PRS_LowRisk'] = personalized_risk_scores[:, 1]
    sliced_data['PRS_MediumRisk'] = personalized_risk_scores[:, 2]

    # Apply the function to create a new column 'PersonalizedRiskScore'
    sliced_data['PersonalizedRiskScore'] = sliced_data.apply(
        lambda row: get_prs(
            row['RiskCategory'],
            row['PRS_HighRisk'],
            row['PRS_LowRisk'],
            row['PRS_MediumRisk']
        ),
        axis=1
    )

    # Drop the individual PRS columns
    sliced_data.drop(columns=['PRS_HighRisk', 'PRS_LowRisk', 'PRS_MediumRisk'], inplace=True)

    st.subheader("Predict Risk Category and Personalized Risk Score")

    # Input fields on the left (col1)
    input_data = {}
    for col in filtered_columns:
        if col in categorical_cols:
            unique_values = le[col].classes_  # Use classes_ to get unique values for the dropdown
            description = f"Select {col}: {unique_values.tolist()}"
            input_data[col] = st.selectbox(f"Select {col}", unique_values, help=description)
        else:
            description = f"Enter {col}. Mean value is {df[col].mean()}"
            input_data[col] = st.number_input(f"Enter {col}", value=float(df[col].mean()), help=description)

    # Convert input data to dataframe for prediction
    input_df = pd.DataFrame([input_data])

    # Encode categorical variables in input data
    for col in categorical_cols:
        input_df[col] = le[col].transform(input_df[col])

    # Predict the risk category and the probabilities for the risk scores
    risk_category_encoded = rf_model.predict(input_df)[0]
    risk_category = le_target.inverse_transform([risk_category_encoded])[0]
    risk_probabilities = rf_model.predict_proba(input_df)[0]

    # Map the predicted probabilities to the appropriate risk category
    personalized_risk_score = get_prs(risk_category, risk_probabilities[0], risk_probabilities[1], risk_probabilities[2])

    # Display predictions in a separate expander section
    with st.expander("Prediction Results"):
        st.markdown(f"<h2 style='text-align: center; font-weight: bold;'>Predicted Risk Category</h2>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center; font-size: 2.5em; font-weight: bold;'>{risk_category}</h1>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align: center; font-weight: bold;'>Personalized Risk Score</h2>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center; font-size: 3em; font-weight: bold; color: red;'>{personalized_risk_score:.2f}</h1>", unsafe_allow_html=True)

    # Provide an overview of the uploaded data in an expander
    with st.expander("Data Overview"):
        st.write(df.describe())

    # Feature descriptions in a separate expander
    feature_descriptions = {
        'Term': "The length of the loan in months (12, 36, or 60).",
        'LoanStatus': "The current status of the loan (e.g., Current, Completed, Defaulted).",
        'BorrowerAPR': "The Annual Percentage Rate of the loan.",
        'EstimatedReturn': "The estimated return on the loan.",
        'ProsperRating (numeric)': "The Prosper rating for the loan, from 1 (high risk) to 7 (low risk).",
        'ListingCategory (numeric)': "The category of the loan purpose (e.g., debt consolidation, business, etc.).",
        'BorrowerState': "The state where the borrower resides.",
        'Occupation': "The occupation of the borrower.",
        'EmploymentStatus': "The employment status of the borrower.",
        'IsBorrowerHomeowner': "Whether the borrower owns a home (True/False).",
        'CreditScoreRangeLower': "The lower bound of the borrower's credit score range.",
        'OpenCreditLines': "The number of open credit lines the borrower has.",
        'TotalCreditLinespast7years': "The total number of credit lines the borrower has had in the past 7 years.",
        'OpenRevolvingAccounts': "The number of open revolving accounts the borrower has.",
        'OpenRevolvingMonthlyPayment': "The monthly payment on the borrower's open revolving accounts.",
        'InquiriesLast6Months': "The number of credit inquiries the borrower has made in the last 6 months.",
        'CurrentDelinquencies': "The number of current delinquencies on the borrower's credit report.",
        'AmountDelinquent': "The total amount that the borrower is delinquent on.",
        'PublicRecordsLast10Years': "The number of public records (e.g., bankruptcies, liens) in the last 10 years.",
        'RevolvingCreditBalance': "The total balance on the borrower's revolving credit accounts.",
        'BankcardUtilization': "The borrower's credit card utilization rate.",
        'DebtToIncomeRatio': "The borrower's debt-to-income ratio.",
        'IncomeRange': "The borrower's income range.",
        'IncomeVerifiable': "Whether the borrower's income is verifiable (True/False).",
        'StatedMonthlyIncome': "The borrower's stated monthly income.",
        'LoanOriginalAmount': "The original amount of the loan.",
        'MonthlyLoanPayment': "The monthly payment on the loan."
    }

    with st.expander("Feature Descriptions"):
        for feature, description in feature_descriptions.items():
            st.markdown(f"**{feature}**: {description}")
