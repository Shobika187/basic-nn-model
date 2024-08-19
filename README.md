# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Shobika P
### Register Number: 212221230096

### Importing Required packages

```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from google.colab import auth
import gspread
from google.auth import default
```
### Authenticate the Google sheet
```
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet=gc.open('Untitled spreadsheet').sheet1
data=worksheet.get_all_values()

dataset1=pd.DataFrame(data[1:],columns=data[0])
dataset1=dataset1.astype({'Input':float})
dataset1=dataset1.astype({'Output':float})
dataset1.head()
X=dataset1[['Input']].values
y=dataset1[['Output']].values
```
### Split the testing and training data
```
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=33)
Scaler=MinMaxScaler()
Scaler.fit(X_train)
X_train1=Scaler.transform(X_train)

```
### Build the Deep learning Model
```
ai_brain=Sequential([
     Dense(units=8,activation='relu'),
     Dense(units=10,activation='relu'),
     Dense(1)
])

ai_brain.compile(optimizer='adam',loss='mse')
ai_brain.fit(X_train1,y_train,epochs=2000)

loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()
```
### Evaluate the Model
```
loss_df=pd.DataFrame(ai_brain.history.history)
loss_df.plot()
X_test1=Scaler.transform(X_test)
ai_brain.evaluate(X_test1,y_test)
X_n1=[[4]]
X_n1=Scaler.transform(X_n1)
ai_brain.predict(X_n1)

```



## OUTPUT
## Dataset Information
![image](https://github.com/user-attachments/assets/6cee3260-0919-4a27-a0dc-1660dc92f0a9)


### Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/b99d4561-0766-4971-b472-20ac4caac33e)


### Test Data Root Mean Squared Error

![image](https://github.com/user-attachments/assets/e9260b06-19f9-4220-a80a-40abb831a16c)


### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/f95cd2ea-a619-4d0b-8884-a3516ec6778c)


## RESULT

Thus a Neural network for Regression model is Implemented
