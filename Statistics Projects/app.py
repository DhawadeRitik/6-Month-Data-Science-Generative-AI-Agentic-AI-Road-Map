import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import scipy.stats as stats
import streamlit as st 


# create a dataframe 
def generate_data():
    np.random.seed(42)
    data = {
    'product_id':range(1,21),
    'product_name':[f'Product {i}' for i in range(1,21)],
    'category':np.random.choice(['Electronics','Clothing','Home','Sports'],20),
    'unit_sold':np.random.poisson(lam=20,size=20), 
    'sale_date':pd.date_range(start='2023-01-01',periods=20,freq='D')  }
    return pd.DataFrame(data)
     
sale_data = generate_data()

# Display the sale data 
st.subheader("Sales Data")
st.dataframe(sale_data) 

# Descriptive summary of datframe 
st.subheader("Descriptive Statistics")
desc_stats = sale_data['unit_sold'].describe()
st.write(desc_stats)

# Additional Summary of the unit sold 
mean = sale_data['unit_sold'].mean()
median =  sale_data['unit_sold'].median()
mode = sale_data['unit_sold'].mode()[0]

st.write(f'Mean of unit sold : {mean}')
st.write(f'median of unit sold :{median}')
st.write(f'Mode of unit sold : {mode}')

# Perform aggregation by category 
category = sale_data.groupby('category')['unit_sold'].agg(['sum','mean','std']).reset_index()
category.columns=['Category','Total Unit Sold','Average Unit Sold','Std od unit sold']
st.subheader('Statistical Category')
st.dataframe(category)

# Inferential Statistics 
# calculate the confidance interval 
confidence_level = 0.95
degree_freedom = len(sale_data['unit_sold']) - 1
sample_mean = sale_data['unit_sold'].mean()
std = sale_data['unit_sold'].std()
standard_error = std / np.sqrt(len(sale_data['unit_sold']))
t_score = stats.t.ppf((1+confidence_level)/2,degree_freedom)
margin_error = t_score * standard_error
confidance_interval = (sample_mean - margin_error, sample_mean + margin_error)
st.subheader("Confidence Interval")
st.write(confidance_interval)

# hypothesis testing 
t_stats , p_value = stats.ttest_1samp(sale_data['unit_sold'],20)
st.subheader("Hypothesis Testing")
st.write(f't stats value : {t_stats}')
st.write(f'p value : {p_value}')

if p_value < 0.05:
    st.write('Reject the null hypothesis')
else :
    st.write('Fail to reject the null hypothesis.')

# Visualization 
st.subheader("Visualization")
plt.figure(figsize=(10,5))
sns.histplot(sale_data['unit_sold'],bins=10,kde=True)
plt.xlabel('Unit Sold')
plt.ylabel("Frequency")
plt.axvline(mean,color='red',linestyle='--',label='mean')
plt.axvline(median,color='green',linestyle='--',label='median')
plt.axvline(mode,color='orange',linestyle='--',label='mode')
plt.legend()
st.pyplot(plt)

# boxplot 
plt.figure(figsize=(10, 6))
sns.boxplot(x='category', y='unit_sold', data=sale_data)
plt.title('Boxplot of Units Sold by Category')
plt.xlabel('Category')
plt.ylabel('Units Sold')
st.pyplot(plt)

# bar plot 
plt.figure(figsize=(10,5))
ax = sns.barplot(data=category,x='Category',y='Total Unit Sold')
plt.xlabel('Category')
plt.ylabel('Unit Sold')
plt.title("Barplot : Category wise Unit sold")
for label in ax.containers:
    ax.bar_label(label)
st.pyplot(plt)