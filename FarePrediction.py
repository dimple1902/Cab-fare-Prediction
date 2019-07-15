#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from fancyimpute import KNN
from ggplot import *
import warnings
import pickle


# In[ ]:


# To remove warnings
warnings.simplefilter('ignore')


# In[ ]:


working_dir= input("Please Enter wrking Directory: ")
os.chdir (working_dir)


# In[ ]:


#read  csv data and save in a variable
train_file =  input("Please enter train data file path(CSV): ")
test_file =  input("Please enter test data file path(CSV): ")
train = pd.read_csv(train_file)
test = pd.read_csv(test_file)


# In[ ]:


train.shape


# In[ ]:


test.shape


# # Data Cleanup Process
#Convert datatypes for both train and test data
#----------------pickup_datetime --> year/month/date/weekday/hourn
#----------------pickup lat/long and dropoff lat/long -->distance
# In[ ]:


train["fare_amount"] = pd.to_numeric(train["fare_amount"], errors = "coerce")
for i in [train,test]:
    i["pickup_datetime"] = pd.to_datetime(i["pickup_datetime"], errors ="coerce")
    
    i["year"] = i["pickup_datetime"].dt.year
    i["year"] = (i["year"]).astype("object")
    i["month"] = i["pickup_datetime"].dt.month
    i["month"] = (i["month"]).astype("object")
    i["day"] = i["pickup_datetime"].dt.day
    i["day"] = (i["day"]).astype("object")
    i["hour"] = i["pickup_datetime"].dt.hour
    i["hour"] = (i["hour"]).astype("object")
    i["weekday"] = i["pickup_datetime"].dt.dayofweek
    i["weekday"] = (i["weekday"]).astype("object")
    i["pickup_datetime"] = i["pickup_datetime"].astype('str')


# In[ ]:


# calculate distance between pickup and dropoff
def havrsine_distance(lat1, long1, lat2, long2):
    data = [train, test]
    for i in data:
        R = 6371  #radius of earth in kilometers
        phi1 = np.radians(i[lat1])
        phi2 = np.radians(i[lat2])
    
        delta_phi = np.radians(i[lat2]-i[lat1])
        delta_lambda = np.radians(i[long2]-i[long1])
    
        a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
        d = (R * c) #in kilometers
        i['distance'] = d


# In[ ]:


havrsine_distance('pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')


# In[ ]:


# re order columns, and put fare_amount in last column as this is the target variable 
train = train.reindex(list(train.columns[1:])+["fare_amount"],axis = 1)


# In[ ]:


train.dtypes


# In[ ]:


test.dtypes


# In[ ]:


train.describe()

#------------------------Check Anamolies and remove them-------------------------------------
# In[ ]:


#There are some anamolies we need to remove
#Longitude valid range  be +/-180 degree
#Latitude range is  +/- 90, but in pickup latitude max is 401.08 which is invalid
#Passenger count minimum is 0 which is not possible and maximum is 5345 which is also unreal
#so lets assume maximum passenger count be 20 ( lets assume cab can be  is a mini bus too)
#fare amount cannot be negative
#So we need to remove these anomalies


train =  train[((train['pickup_longitude'] > -180) & (train['pickup_longitude'] < 180)) & 
               ((train['dropoff_longitude'] > -180) & (train['dropoff_longitude'] < 180)) & 
               ((train['pickup_latitude'] > -90) & (train['pickup_latitude'] < 90)) & 
               ((train['dropoff_latitude'] > -90) & (train['dropoff_latitude'] < 90)) & 
               (train['passenger_count'] > 0) & (train['passenger_count'] <=20) & (train['fare_amount']>0)]


# In[ ]:


train.shape


# In[ ]:


train.describe()


# In[ ]:


numerical = ['pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'distance','fare_amount']


# In[ ]:


categorical = ['pickup_datetime', 'year',
       'month', 'day', 'hour', 'weekday']


# In[ ]:


# Outliers in Numerical Variables
#boxplot to visualise outlier
#get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(train["distance"])


# In[ ]:


# we can see there are some outliers in distance
# lets count outliers if they are less then remove them
train.loc[(train["distance"]>200),"distance"].count()
# the outliers are only 23 out of 15906 so we can remove it


# In[ ]:


#Remove distance outliers
train = train.drop(train.loc[(train["distance"]>200),:].index, axis=0)


# In[ ]:


train.shape


# In[ ]:


plt.boxplot(train["fare_amount"])


# In[ ]:


# we can see there are some outliers in farre_amount
# lets count outliers if they are less then remove them
train.loc[(train["fare_amount"]>200),"fare_amount"].count()
# the outliers are only 4 out of 15883 so we can remove it


# In[ ]:


#Remove outliers in fare_amount
train = train.drop(train.loc[(train["fare_amount"]>200),:].index, axis=0)


# In[ ]:


#scatter plot between distance and fare amount 
from ggplot import *

ggplot( aes(x = 'distance', y = "fare_amount"),data = train) + geom_point()


# In[ ]:


# In scatter plot we can see that at some points where distance is 0, fare is not 0 so we need  count these anamolies
train.loc[(train["distance"]==0)&train["fare_amount"]>0,"distance"].count()
#this is 454 and a 2% of total data so we can remove them as well


# In[ ]:


train = train.drop(train.loc[((train["distance"]==0)&train["fare_amount"]>0),:].index, axis=0)


# In[ ]:


train.shape


# In[ ]:


# similarly if fare is 0 then distance cannot be >0
train.loc[(train["fare_amount"]==0)&train["distance"]>0,"fare_amount"].count()


# In[ ]:


#check observations when fare and distance both 0,
train.loc[(train["fare_amount"]==0)&(train["distance"]==0),"fare_amount"].count()


# In[ ]:


#Remove rows where passenger count is 0
train = train.drop(train.loc[(train["passenger_count"]==0),:].index, axis=0)


# In[ ]:


train.shape

#----------------------------------Missing data Analysis------------------------
# In[ ]:


missing_val = pd.DataFrame(train.isnull().sum())


# In[ ]:


missing_val


# In[ ]:


# there is only one row which has null value So we can remove it
train  = train.drop(train.loc[(train["year"].isnull()),:].index, axis=0)


# In[ ]:


#We can remove pickup_datetime as it doesnt have much information in general
train = train.drop(['pickup_datetime'], axis = 1)
test = test.drop(['pickup_datetime'], axis = 1)


# In[ ]:


train.shape


# # Work on given test data variables

# In[ ]:


test.dtypes


# In[ ]:


train.dtypes


# In[ ]:


train.loc[:,"passenger_count"] = train.loc[:,"passenger_count"].astype('int64')


# In[ ]:


train.describe()


# In[ ]:


test.describe()


# In[ ]:


train.to_csv("processed_train_data.csv")


# # Visualaization and  Feature Selection

# In[ ]:


##Correlation analysis As all data is Numerical
#Assume that there should be no dependency between independent variables 
#Assume that there should be high dependency between independent variables and independent variable
#Correlation plot
train_corr = train.loc[:,numerical]


# In[ ]:


#Set the width and hieght of the plot
f, ax = plt.subplots(figsize=(7, 5))

#Generate correlation matrix
corr = train_corr.corr()

#Plot using seaborn library
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


# In[ ]:


train.shape


# In[ ]:


#we can remove all the geolocation points as they are highly corelated and distance is actually derived from these variables
# Also we can see that fare amount depends on week, day and date as desplayed in following scatter graphs

train = train.drop(['pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude'], axis = 1)
test = test.drop(['pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude'], axis = 1)


# In[ ]:



# How week days affect fare_amount
ggplot(train, aes(x = 'weekday', y = 'fare_amount', color='weekday')) +     geom_point(alpha = 1, size = 50) + theme_bw()+ ylab("fare_amount") + xlab("weekday") + ggtitle("Scatter Plot Analysis weekday vs fare")
# maximum fare is on friday, Monday, Tuesday, wednesday and low fares on Sunday 


# In[ ]:


#Impact of date on fare_amount
ggplot(train, aes(x = 'day', y = 'fare_amount', color='weekday')) +     geom_point(alpha = 1, size = 50) + theme_bw()+ ylab("fare_amount") + xlab("date") + ggtitle("Scatter Plot Analysis date vs fare")
# If fare is affected by date Maximum is around 7th but fare is almost same on all days so date doesnt affect fare amount


# In[ ]:


#Impact of hour on fare_amount
ggplot(train, aes(x = 'hour', y = 'fare_amount', color='weekday')) +     geom_point(alpha = 1, size = 50) + theme_bw()+ ylab("fare_amount") + xlab("hour") + ggtitle("Scatter Plot Analysis hour vs fare")
# we can see that fare amount is less in the nights sometimes but mostly alomost same


# In[ ]:


# impact of passenger count on fare
ggplot(train, aes(x = 'passenger_count', y = 'fare_amount', color='weekday')) +     geom_point(alpha = 1, size = 50) + theme_bw()+ ylab("Fare") + xlab("Passenger Count") + ggtitle("Scatter Plot Analysis passenger count vs fare")

# If fare is affected by passenger count, highest when passenger count is 1-2


# In[ ]:


# distance impact on fare amount 
ggplot(train, aes(x = 'distance', y = 'fare_amount',color = 'weekday')) +     geom_point(alpha = 1, size = 50) + theme_bw()+ ylab("Fare") + xlab("Distance") + ggtitle("Scatter Plot Analysis distance vs fare")
#We can see thta more the distane more the fare, so clearly fare amount is highly positively corelated with distance


# In[ ]:


train.dtypes


# In[ ]:


test.dtypes


# In[ ]:


#train = df.copy()


# In[ ]:


train.columns


# In[ ]:


test.columns


# # Feature Scaling

# In[ ]:


#feature scaling is always for continuous variables
#Normality check
ggplot(train, aes(x = 'passenger_count')) + geom_histogram(fill="DarkSlateBlue", colour = "black") +    geom_density() +    theme_bw() + xlab("passenger_count") + ylab("Frequency") + ggtitle("Passenger Count Analysis") +    theme(text=element_text(size=20))


# In[ ]:



ggplot(train, aes(x = 'distance')) + geom_histogram(fill="Green", colour = "black") +    geom_density() +    theme_bw() + xlab("distance") + ylab("Frequency") + ggtitle("Distance Normality Analysis") +    theme(text=element_text(size=20))


# In[ ]:



ggplot(train, aes(x = 'fare_amount')) + geom_histogram(fill="Orange", colour = "black") +    geom_density() +    theme_bw() + xlab("fare_amount") + ylab("Frequency") + ggtitle("Fare Amount Analysis") +    theme(text=element_text(size=20))


# In[ ]:


train.columns


# In[ ]:


#As the data is not equally distributed so we will go with Nomalisation
for i in ['passenger_count', 'distance']:
    print(i)
    train[i] = (train[i] - min(train[i]))/(max(train[i]) - min(train[i]))
    test[i] = (test[i] - min(test[i]))/(max(test[i]) - min(test[i]))


# In[ ]:


train.head()


# # Model development

# # Linear Regression

# In[ ]:


x = train.copy()
#train = x.copy()
# we need to sample train data again in test and train so that we can apply model on sample 
from sklearn.model_selection import train_test_split
train,test1 = train_test_split(train,test_size=0.15)


# In[ ]:


train.shape


# In[ ]:


# for linear regression we need continuous variables so convert  all the data in numeric
train_linear = train.copy()
test1_linear = test1.copy()
for j in [train_linear,test1_linear]:
    for i in range(0, j.shape[1]):
        if(j.iloc[:,i].dtypes == "object"):
            j.iloc[:,i] = pd.to_numeric(j.iloc[:,i])


# In[ ]:


#First start with Linear Regression
#Import Libraries for LR
#import models for libear regression
# As  the target v
import statsmodels.api as sm


# In[ ]:


test1.shape


# In[ ]:


#train models using the training sets (optimum least swuare method to calculate coefficient)
#parameters dependent variable , independent variables
lmodel = sm.OLS(train_linear.iloc[:,7], train_linear.iloc[:,0:7]).fit()


# In[ ]:


lmodel.summary()
#rSquared 79.7%  dependent variable can be is explained by independent variables


# In[ ]:


lr_prediction = lmodel.predict(test1_linear.iloc[:,0:7])


# In[ ]:


#Calculate MAPE
def mape(y_true,y_pred):
    #print(np.mean(abs((y_true-y_pred)/y_true)))*100
    return 'Test MAPE : %.3f' % (np.mean(abs((y_true-y_pred)/y_true))*100)


# In[ ]:


#Caclulate RMSE
from sklearn.metrics import mean_squared_error
def RMSE (y_true,y_pred):
    return("Test RMSE: %.3f" % mean_squared_error(y_true, y_pred) ** 0.5)


# In[ ]:

print("\nLinear Regression Model")
print(mape(test1_linear.iloc[:,7],lr_prediction))


# In[ ]:


print(RMSE(test1_linear.iloc[:,7],lr_prediction))


# In[ ]:


#train = x.copy()


# # Decision Regression Tree

# In[ ]:


#decision Tree for regression (max_depth = 2 means max branch for any node should be 2)
from sklearn.tree import DecisionTreeRegressor
DT_regression = DecisionTreeRegressor(max_depth = 7).fit(train.iloc[:,0:7],train.iloc[:,7])


# In[ ]:


#apply Data model on Test
DT_prediction = DT_regression.predict(test1.iloc[:,0:7])


# In[ ]:

print("\nDecision Tree Algo")
print(mape(test1.iloc[:,7],DT_prediction))


# In[ ]:


print(RMSE(test1.iloc[:,7],DT_prediction))


# # Random forest

# In[ ]:


#Random Forest
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


RF_Model = RandomForestRegressor(n_estimators=150).fit(train.iloc[:,0:7],train.iloc[:,7])


# In[ ]:


RF_prediction = RF_Model.predict(test1.iloc[:,0:7])


# In[ ]:

print("\nRandom Forest Regressor")
print(mape(test1.iloc[:,7],RF_prediction))


# In[ ]:


print(RMSE(test1.iloc[:,7],RF_prediction))


# # KNN ML Algo
# 

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
KN_Model = KNeighborsRegressor(n_neighbors=60).fit(train.iloc[:,0:7],train.iloc[:,7])


# In[ ]:


KN_prediction = KN_Model.predict(test1.iloc[:,0:7])


# In[ ]:

print("\nKNN Regressor")
print(mape(test1.iloc[:,7],KN_prediction))


# In[ ]:


print(RMSE(test1.iloc[:,7],KN_prediction))


# # Best Predictor is Random Decision Tree so predict actual test data using DT and save it in CSV

# In[ ]:


test["fare_amount"]=0


# In[ ]:


test.shape


# In[ ]:


final_test =pd.DataFrame(RF_Model.predict(test.iloc[:,0:7]))


# In[ ]:


test["fare_amount"]=final_test[0]


# In[ ]:


test


# In[ ]:


final_test.head()


# In[ ]:


test_submission = pd.read_csv("test.csv")


# In[ ]:


test_submission["fare_amount"] = final_test


# In[ ]:


output_predict_submission = input("Please enter path of final test submission file(csv): ")


# In[ ]:


test_submission.to_csv(output_predict_submission)


# In[ ]:




