import streamlit as st 
import pickle as pkl
import numpy as np


st.title('Salary Prediction App')
# load the save model
model = pkl.load(open(r'D:\NIT Course\NIT Data Science Course\Machine Learning\LinearRegression\multiple_linear_regression\SLR - House price prediction\backend_code.pkl','rb'))

# Add a brief description
st.write("This app predicts the salary based on years of experience using a simple linear regression model.")

# Add input widget for user to enter years of experience
years_experience = st.number_input("Enter Years of Experience:", min_value=0.0, max_value=50.0, value=1.0, step=0.5)

if st.button('Predict Salary'):
    # make a prediction using trained model 
    experience_input = np.array([[years_experience]]) # convert the years experience into the 2d array for prediction
    predicted_salary = model.predict(experience_input) # predict the salary 
    
    # print the predicted salary 
    st.success(f'The predicted Salary for {years_experience} year of experience is : ${predicted_salary[0]:,.2f} ')
    
# Display information about the model
st.write("The model was trained using a dataset of salaries and years of experience.")
