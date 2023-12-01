import streamlit as st
import joblib

# Load the model
load_model = joblib.load('c:/Users/HomePC/Documents/School/Moringa/Phase 3/Project/Phase-3-Project/meta_learner.joblib')

# Streamlit app
def main():
    st.title('SYRIATEL NEW SUBSCRIBER CHURN PREDICTOR')

    # Example: Taking input from user
    international_plan = st.selectbox('International Plan', ['Yes', 'No'])
    voice_mail_plan = st.selectbox('Voice Mail Plan', ['Yes', 'No'])
    total_day_charge = st.number_input('Total Day Charge', value=0.0, step=0.01)
    total_intl_calls = st.number_input('Total Intl Calls', value=0, step=1)
    total_intl_charge = st.number_input('Total Intl Charge', value=0.0, step=0.01)
    customer_service_calls = st.number_input('Customer Service Calls', value=0, step=1)
    day_charge_minute_ratio = st.number_input('Day Charge Minute Ratio', value=0.0, step=0.01, max_value=1.0)
    intl_charge_minute_ratio = st.number_input('Intl Charge Minute Ratio', value=0.0, step=0.01, max_value=1.0)
    state_target_encoded = st.number_input('State Target Encoded', value=0.0, step=0.01, max_value=1.0)

    # Fill missing values with defaults
    default_values = [0.0, 0, 0.0, 0, 0.0]

    # Prepare input data for prediction
    input_data = [[international_plan == 'Yes', voice_mail_plan == 'Yes', total_day_charge,
                   total_intl_calls, total_intl_charge, customer_service_calls,
                   day_charge_minute_ratio, intl_charge_minute_ratio,
                   state_target_encoded] + default_values]

    # Make predictions based on user input
    if st.button('Predict'):
        # Make predictions
        prediction = load_model.predict(input_data)

        # Show prediction result
        st.write(f'The prediction is: {prediction[0]}')

if __name__ == '__main__':
    main()
