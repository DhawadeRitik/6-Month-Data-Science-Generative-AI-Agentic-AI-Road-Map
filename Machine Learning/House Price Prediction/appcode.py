import streamlit as st 
import pickle as pkl 
import numpy as np


st.title("House Price Prediction App")

# name of the used models
model_names = [
'Linear Regression', 'HuberRegressor', 'Ridge', 'Lasso', 'Elastic Net Regression',
    'PolynomialRegression',  'ANN Regression', 'Random Forest', 'SVR Regression', 'LGBM',
    'XGBoost', 'KNeighbors Regressor'
]

# dump the model stored file 
models = {name: pkl.load(open(f'{name}.pkl', 'rb')) for name in model_names}

model_used = {
    'Linear Regression' :  'Linear Regression', 
    'HuberRegressor' : 'HuberRegressor', 
    'Ridge' :  'Ridge' , 
    'Lasso' :'Lasso', 
    'Elastic Net Regression' : 'Elastic Net Regression',
    'PolynomialRegression' :  'PolynomialRegression', 
    'ANN Regression' : 'ANN Regression', 
    'Random Forest' : 'Random Forest' , 
    'SVR Regression' : 'SVR Regression', 
    'LGBM' : 'LGBM',
    'XGBoost' : 'XGBoost', 
    'KNeighbors Regressor' :  'KNeighbors Regressor'
}

# create select box from which it can choose an option of the models  
options = st.selectbox(
    "Select the model",
    list(model_used.keys())
)

## Input the user need to give 
Avg_Area_Income = st.number_input(" Choose Avg. Area Income",min_value=  17797, max_value=104703)
House_Age = st.number_input(" Choose Avg. Area House Age",min_value=2, max_value=10)
Number_Rooms = st.number_input(" Choose Avg. Area Number of Rooms", min_value=3, max_value=11)
Number_Bedrooms = st.number_input(" Choose Avg. Area Number of Bedrooms",min_value=2, max_value=7)
Area_Population = st.number_input(" Choose Avg. Area Population",min_value=173, max_value=69622)

# create a button to sent the input to models
if st.button("Predict Price"):
    if not all([Avg_Area_Income,House_Age,Number_Rooms,Number_Bedrooms,Area_Population]):
        st.warning("Please enter all input fields")
    else :
        input_array = np.array([[Avg_Area_Income,House_Age,Number_Rooms,Number_Bedrooms,Area_Population]])
        selected_model = model_used[options]
        model_used = models[selected_model]
        
        # predict the price 
        predicted_price = model_used.predict(input_array)[0]
        st.success(f"Predicted House Price for {options}  is ${predicted_price:,.2f}:")
 
st.markdown("Note :")       
st.markdown("This Model are trained using the USA Housing Data Set")
        
