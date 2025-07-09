import pandas as pd 
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle as pkl
from sklearn.metrics import mean_squared_error


df = pd.read_csv(r"D:\NIT Course\NIT Data Science Course\Machine Learning\LinearRegression\Salary_Data - Salary_Data.csv")
X = df[['YearsExperience']]
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)

slr = LinearRegression()
slr.fit(X_train, y_train)
y_pred = slr.predict(X_test) 

# comparition table 
comparision  = pd.DataFrame({"Actual Salary":y_test,"Predicted Salary":y_pred})


# visualization of training dataset 
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,slr.predict(X_train),color='green')
plt.title('Visualization : Training Data ')
plt.xlabel('Year Of Experience')
plt.ylabel('Salary')



# visualization of testing data
plt.scatter(X_test,y_test,color='green')
plt.plot(X_test,y_pred,color='red')
plt.title('Visualization : Test Data set')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')



# visualization of actual vs predicted data points 
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, color='blue', label='Actual Salary')
plt.scatter(X_test, y_pred, color='orange', label='Predicted Salary', marker='x')
plt.plot(X_test,y_pred,color='magenta')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Actual vs Predicted Salaries")
plt.legend()



train_mse = mean_squared_error(y_train, slr.predict(X_train))
test_mse = mean_squared_error(y_test, y_pred)
bias = slr.score(X_train,y_train)
variance = slr.score(X_test,y_test)

# save the train model to disk 
filename = 'backend_code.pkl'
with open(filename , 'wb') as file :
    pkl.dump(slr, file)
print('Model has been pickled and saved as backend_code.pkl')
    