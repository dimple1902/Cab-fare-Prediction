rm(list=ls(all=T))
#Load Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees','scales','psych','gplots')


#install.packages(x)
lapply(x, require, character.only = TRUE)
rm(x)

#set working Directory
setwd('D:/edwisor/CabFarePrediction')

#import Test data and train data
train = read.csv("train_cab.csv") 

test = read.csv("test.csv")
str(train)
str(test)


#Data CleanUp
#Convert datatypes for both train and test data
#----------------pickup_datetime --> year/month/date/weekday/hourn
#----------------pickup lat/long and dropoff lat/long -->distance
#train$pickup_date =as.Date(train$pickup_datetime)
library(anytime)
library(lubridate)


train$pickup_datetime =anytime(train$pickup_datetime)
train$year = year(train$pickup_datetime)
train$month = months(train$pickup_datetime)
train$date = day(train$pickup_datetime)
train$weekday = weekdays(train$pickup_datetime)
train$hour = hour(train$pickup_datetime)
#train$fare_amount = as.numeric(train$fare_amount)

#Similarly for test data
test$pickup_datetime =anytime(test$pickup_datetime)
test$year = year(test$pickup_datetime)
test$month = months(test$pickup_datetime)
test$date = day(test$pickup_datetime)
test$weekday = weekdays(test$pickup_datetime)
test$hour = hour(test$pickup_datetime)
   
library(NISTunits)

#Haversine distance to calculate distance between two geo locations
Haversine_dist = function(df,lat1, long1, lat2, long2){
    R = 6371  #radius of earth in kilometers
    
    pi1 = NISTdegTOradian(df[[lat1]])
    pi2 = NISTdegTOradian(df[[lat2]])
    
    delta_pi = NISTdegTOradian(df[[lat2]]-df[[lat1]])
    delta_lambda = NISTdegTOradian(df[[long2]]-df[[long1]])
    
    a = sin(delta_pi / 2.0) ** 2 + cos(pi1) * cos(pi2) * sin(delta_lambda / 2.0) ** 2
    
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    d = (R * c) #in kilometers
    df[["distance"]]= d
    return (df)
    
  }


train = Haversine_dist(train, 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')
test = Haversine_dist(test, 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')


# re order columns, and put fare_amount in last column as this is the target variable

fareamount = data.frame(train$fare_amount)
train = train[,-1]
train$fare_amount =fareamount[,1]
str(train)
str(test)

#There are some anamolies we need to remove
#Longitude valid range  be +/-180 degree
#Latitude range is  +/- 90, but in pickup latitude max is 401.08 which is invalid
#Passenger count minimum is 0 which is not possible and maximum is 5345 which is also unreal
#so lets assume maximum passenger count be 20 ( lets assume cab can be  is a mini bus too)
#fare amount cannot be negative
#So we need to remove these anomalies
train = train[(train$pickup_longitude>-180 & train$pickup_longitude<180),]
train = train[(train$dropoff_longitude>-180 & train$dropoff_longitude<180),]
train = train[(train$pickup_latitude>-90 & train$pickup_latitude<90),]
train = train[(train$dropoff_latitude>-90 & train$dropoff_latitude<90),]
train = train[(train$passenger_count>0 & train$passenger_count<20),]
train = train[(train$distance >00 & train$distance<300),]
train = train[(as.numeric(train$fare_amount)>0),]

#remove pickup_datetime as we have already parsed it
train = train[,!names(train) %in% c("pickup_datetime")]
test = test[,!names(test) %in% c("pickup_datetime")]

#Checking missing values
missing_val = data.frame(apply(train,2,function(x){sum(is.na(x))}))
missing_val$Columns = row.names(missing_val)
names(missing_val)[1] =  "Missing_percentage"
missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(train)) * 100
missing_val = missing_val[order(-missing_val$Missing_percentage),]
row.names(missing_val) = NULL
missing_val = missing_val[,c(2,1)]
write.csv(missing_val, "Miising_perc.csv", row.names = F)
train_bk =train
str(train)
train$fare_amount = as.numeric(as.character( train$fare_amount))

train$weekday = as.numeric(as.factor(train$weekday))
train$month=as.numeric(as.factor(train$month))

#actual --0.8511597
#Mean -->  3.442187
#Median -->2.201407
#KNN --> 8.029433
#train[7000,"distance"]
#train[7000,"distance"] = NA

#Mean Method
#train$distance[is.na(train$distance)] = mean(train$distance, na.rm = T)
#Median Method
for (i in colnames(train)){
  train[is.na(train[,i]),i] = median(train[,i],na.rm = T)
  
}
train$distance[is.na(train$distance)] = median(train$distance, na.rm = T)

#KNN Method

#train = knnImputation(train, k = 3)
sum(is.na(train))


#train = train_bk

#checking outliers
numeric_index = sapply(train,is.numeric) #selecting only numeric

numeric_data = train[,numeric_index]

cnames = colnames(numeric_data)

for (i in 1:length(cnames))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i])), data = subset(train))+
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=cnames[i])+
           ggtitle(paste("Box plot for",cnames[i])))
}

## Plotting plots together
gridExtra::grid.arrange(gn1,gn2,gn3,gn11,ncol=4)
gridExtra::grid.arrange(gn4,gn5,gn6,gn12,ncol=4)
gridExtra::grid.arrange(gn7,gn8,gn9,gn10,ncol=4)
#wE only need to remove fare amount which is very high
train = train[((train$fare_amount)<250),]

#Scatter plot between distance and fare
ggplot(train, aes_string(x = train$distance, y = train$fare_amount)) + 
  geom_point(aes_string(colour = train$weekday),size = 4) +
  theme_bw()+ ylab("fare") + xlab("distance") + ggtitle("Scatter plot Analysis distance vs fare") + 
  theme(text=element_text(size=25))
#longer the distance, more the fare
# In scatter plot we can see that at some points where distance is 0, fare is not 0 , so remove those data points
train = train[!(train$distance==0 & train$fare_amount>0),]
#similarly if fare is 0 then distance cannot be >0
train = train[!(train$distance>0 & train$fare_amount==0),]
#Remove observations when fare and distance both 0,
train = train[!(train$distance==0 & train$fare_amount==0),]

# Work on test data -- 
str(train)
str(test)

test$weekday = as.numeric(as.factor(test$weekday))
test$month=as.numeric(as.factor(test$month))

# Visualaization and Feature Selection
#As all tdata is Numeric, so co relation analyis

## Correlation Plot 
corrgram(train, order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")

# as distance is calculated by geolocations so we can remove geolocations
train = train[,!names(train) %in% c('pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude')]
test = test[,!names(test) %in% c('pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude')]


# How week days affect fare_amount
ggplot(train, aes_string(x = train$weekday, y = train$fare_amount)) + 
  geom_point(aes_string(colour = train$weekday),size = 4) +
  theme_bw()+ ylab("fare") + xlab("weekdy") + ggtitle("Scatter plot Analysis weekday vs fare") + 
  theme(text=element_text(size=25))
# maximum fare is on friday, Monday, Tuesday, wednesday and low fares on Sunday,thursday

# How dates  affect fare_amount
ggplot(train, aes_string(x = train$date, y = train$fare_amount)) + 
  geom_point(aes_string(colour = train$weekday),size = 4) +
  theme_bw()+ ylab("fare") + xlab("date") + ggtitle("Scatter plot Analysis date vs fare") + 
  theme(text=element_text(size=25))

# How month  affect fare_amount
ggplot(train, aes_string(x = train$month, y = train$fare_amount)) + 
  geom_point(aes_string(colour = train$weekday),size = 4) +
  theme_bw()+ ylab("fare") + xlab("month") + ggtitle("Scatter plot Analysis month vs fare") + 
  theme(text=element_text(size=25))

# How hour  affect fare_amount
ggplot(train, aes_string(x = train$hour, y = train$fare_amount)) + 
  geom_point(aes_string(colour = train$weekday),size = 4) +
  theme_bw()+ ylab("fare") + xlab("hour") + ggtitle("Scatter plot Analysis month vs fare") + 
  theme(text=element_text(size=25))
# we can see that fare amount is less in the nights sometimes but mostly alomost same

str(train)
#feature scaling
#we know that day , date, year. month are factored or categorical in nature so we will not do feature scaling for them

#Histogram To check the distribution of continuous variable
ggplot(train, aes_string(x = train$passenger_count)) + 
  geom_histogram(fill="cornsilk", colour = "black",bins = 30) + geom_density() +
  scale_y_continuous(breaks=pretty_breaks(n=10)) + 
  scale_x_continuous(breaks=pretty_breaks(n=10))+
  theme_bw() + xlab("passenger count") + ylab("Frequency") + ggtitle("train: passenger count") +
  theme(text=element_text(size=20))



#As the data is not equally distributed so we will go with Nomalisation
for(i in c('passenger_count', 'distance')){

  train[,i] = (train[,i] - min(train[,i]))/(max(train[,i] - min(train[,i])))
  test[,i] = (test[,i] - min(test[,i]))/(max(test[,i] - min(test[,i])))
}


############################################### Model Development ###################################################################
#Linear Regression
#Divide the data into train and test and test
train.index = sample(nrow(train),size=nrow(train)*0.8,replace = FALSE)
train = train[train.index,]
test1 = train[-train.index,]
#check Multicollinearity --> Variance inflation factor is one of the test
library("usdm")
#variance inflation factor, if vif >10 then colinearity problem
vif(train[,-8])
#To keep the variable or delete it, VIF corelation if 90 % is corelation it is accepted else remove
vifcor(train[,-8],th = 0.9)
#run regression model
lm_model = lm(fare_amount~.,data = train)
summary(lm_model)

prediction_LR = predict(lm_model,test1[,1:7])
mape = function(Ya, Yp){
  mean(abs((Ya-Yp)/Ya))*100
}
mape(test1[,8],prediction_LR) #30.42

#Alternate method rer.val present in DMR library
regr.eval(test1[,8],prediction_LR,stats=c("mse","mae","mape", "rmse"))
#rmse = 4.42

#Decision Regression Tree

#rpart library for regression Decision Tree
DT_Model = rpart(fare_amount~.,data = train, method = "anova")
DT_predictions = predict(DT_Model,test1[-8])
#calculate MAPe

mape(test1[,8],DT_predictions) #26.06

#Alternate method rer.val present in DMR library
regr.eval(test1[,8],DT_predictions,stats=c("mse","mae","mape", "rmse"))
#RMSE --> 3.99

###Random Forest
RF_model = randomForest(fare_amount ~ ., train, importance = TRUE, ntree = 500)


#Presdict test data using random forest model
RF_Predictions = predict(RF_model, test1[,-8])

#calculate MAPe

mape(test1[,8],RF_Predictions)
#] mape = 12.96
#Alternate method rer.val present in DMR library
regr.eval(test1[,8],RF_Predictions,stats=c("mse","mae","mape", "rmse"))
#RMSE --> 1.97

##KNN Implementation
library(class)
#Predict test data
KNN_Predictions = as.numeric(knn(train[, 1:7], test1[, 1:7], train$fare_amount, k = 150))
str(KNN_Predictions)
#calculate MAPe

mape(test1[,8],KNN_Predictions)
#] mape = 299.5046
#Alternate method rer.val present in DMR library
regr.eval(test1[,8],KNN_Predictions,stats=c("mse","mae","mape", "rmse"))
#RMSE -- 25.26

# As we see the best mode is Random Forest so apply the model on test data

test[,"fare_amount"]=0
final_test =data.frame(predict(RF_model, test[,-8]))
test[,"fare_amount"]=final_test
test_submission = read.csv("test.csv")
test_submission[,"fare_amount"] = final_test
write.csv(test_submission,"test_submission_R.csv")

