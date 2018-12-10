# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:43:09 2017

@author: MainPc
"""
#import library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import calendar
from datetime import datetime
from scipy import stats

#read dataset
train = pd.read_csv("c:/train.csv")
test = pd.read_csv("c:/test.csv")

#dataset description
print(train.shape)
print(test.shape)
print(train.head(3))

train.info()
test.info()

#data sidtribution
fig,ax1 = plt.subplots()
fig.set_size_inches(8,5)
sn.boxplot(data=train,y="count",orient="v",ax=ax1)
ax1.set(ylabel='Number of Bike Rental (Hourly)',title="Count")

#remove outliers
newtrain = train[np.abs(train["count"]-train["count"].mean())<=(3*train["count"].std())] 
print ("Shape Before Ouliers: ",train.shape)
print ("Shape After Ouliers: ",newtrain.shape)

#datetime decomposition
newtrain["date"] = newtrain.datetime.apply(lambda x : x.split()[0])
newtrain["hour"] = newtrain.datetime.apply(lambda x : x.split()[1].split(":")[0])
newtrain["weekday"] = newtrain.date.apply(lambda dateString : calendar.day_name[datetime.strptime(dateString,"%Y-%m-%d").weekday()])
newtrain["month"] = newtrain.date.apply(lambda dateString : calendar.month_name[datetime.strptime(dateString,"%Y-%m-%d").month])
newtrain = newtrain.drop(["datetime"],axis=1)

#data visualization
lists = ['hour', 'weekday', 'month']
for i, name in enumerate(lists):
    plt.subplot(3,1,i+1)
    sn.countplot(name,data=newtrain) 
plt.show()

fig,ax2 = plt.subplots()
fig.set_size_inches(8,5)
sn.barplot(data=newtrain,y="count",x="hour",orient="v",ax=ax2)
ax2.set(ylabel='Number of Bike Rental (Hourly) ',title="Hour")

fig,ax3 = plt.subplots()
fig.set_size_inches(8,5)
sn.barplot(data=newtrain,y="count",x="weekday",orient="v",ax=ax3)
ax3.set(ylabel='Number of Bike Rental (Hourly) ',title="Weekday")

fig,ax4 = plt.subplots()
fig.set_size_inches(8,5)
sn.barplot(data=newtrain,y="count",x="month",orient="v",ax=ax4)
ax4.set(ylabel='Number of Bike Rental (Hourly) ',title="Month")

fig,ax5 = plt.subplots()
fig.set_size_inches(8,5)
sn.countplot(newtrain['holiday'],ax=ax5)
ax5.set(xlabel='Holiday', ylabel='Count',title="Holiday")

fig,ax6 = plt.subplots()
fig.set_size_inches(8,5)
sn.countplot(newtrain['workingday'],ax=ax6)
ax6.set(xlabel='Working Day', ylabel='Count',title="Working Day")

fig,ax7 = plt.subplots()
fig.set_size_inches(8,5)
sn.barplot(data=newtrain,y="count",x="workingday",orient="v",ax=ax7)
ax7.set(ylabel='Number of Bike Rental (Hourly) ',title="Working Day")

fig,ax14 = plt.subplots()
fig.set_size_inches(8,5)
sn.boxplot(data=newtrain,y="count",x="weather",orient="v",ax=ax14)
ax14.set(ylabel='Number of Bike Rental (Hourly) ',title="Weather")

fig,(ax8,ax9) = plt.subplots(nrows=2)
fig.set_size_inches(8, 5)
sn.countplot(newtrain['temp'],ax=ax8)
ax8.set( ylabel='Count',title="Temperature")
sn.regplot(x="temp", y="count", data=newtrain,ax=ax9)

fig,(ax10,ax11) = plt.subplots(nrows=2)
fig.set_size_inches(8, 5)
sn.countplot(newtrain['atemp'],ax=ax10)
ax10.set( ylabel='Count',title="Apparent Temperature")
sn.regplot(x="atemp", y="count", data=newtrain,ax=ax11)

fig,(ax11,ax12) = plt.subplots(nrows=2)
fig.set_size_inches(8, 5)
sn.countplot(newtrain['windspeed'],ax=ax11)
ax10.set( ylabel='Count',title="Windspeed")
sn.regplot(x="windspeed", y="count", data=newtrain,ax=ax12)

fig,(ax12,ax13) = plt.subplots(nrows=2)
fig.set_size_inches(8, 5)
sn.countplot(newtrain['humidity'],ax=ax12)
ax12.set( ylabel='Count',title="Humidity")
sn.regplot(x="humidity", y="count", data=newtrain,ax=ax13)

correlation = newtrain[["temp","humidity","windspeed","casual","registered","count"]].corr()
mask = np.array(correlation)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(15,10)
sn.heatmap(correlation, mask=mask,vmax=.8, square=True,annot=True)

fig,ax15 = plt.subplots(ncols=2,nrows=2)
fig.set_size_inches(15, 10)
sn.distplot(train["count"],ax=ax15[0][0])
stats.probplot(train["count"], dist='norm', fit=True, plot=ax15[0][1])
sn.distplot(np.log(newtrain["count"]),ax=ax15[1][0])
stats.probplot(np.log1p(newtrain["count"]), dist='norm', fit=True, plot=ax15[1][1])

fig,month = plt.subplots()
fig.set_size_inches(15,10)
hourAggregated = pd.DataFrame(newtrain.groupby(["hour","month"],sort=True)["count"].mean()).reset_index()
sn.pointplot(x=hourAggregated["hour"], y=hourAggregated["count"],hue=hourAggregated["month"], data=hourAggregated, join=True,ax=month)
month.set(xlabel='Hour Of The Day', ylabel='Number of Bike Rental',title="Nmumber Of Bike Rental By Hour Of The Day Across Month",label='big')

fig,weekday = plt.subplots()
fig.set_size_inches(15,10)
hueOrder = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
hourAggregated = pd.DataFrame(newtrain.groupby(["hour","weekday"],sort=True)["count"].mean()).reset_index()
sn.pointplot(x=hourAggregated["hour"], y=hourAggregated["count"],hue=hourAggregated["weekday"],hue_order=hueOrder, data=hourAggregated, join=True,ax=weekday)
weekday.set(xlabel='Hour Of The Day', ylabel='Number of Bike Rental',title="Nmumber Of Bike Rental By Hour Of The Day Across Weekdays",label='big')

fig,user = plt.subplots()
fig.set_size_inches(15,10)
hourTransformed = pd.melt(newtrain[["hour","casual","registered"]], id_vars=['hour'], value_vars=['casual', 'registered'])
hourAggregated = pd.DataFrame(hourTransformed.groupby(["hour","variable"],sort=True)["value"].mean()).reset_index()
sn.pointplot(x=hourAggregated["hour"], y=hourAggregated["value"],hue=hourAggregated["variable"],hue_order=["casual","registered"], data=hourAggregated, join=True,ax=user)
user.set(xlabel='Hour Of The Day', ylabel='Number of Bike Rental',title="Nmumber Of Bike Rental By Hour Of The Day Across User Type",label='big')

#remove unnecessary features
dataset = train.drop(["datetime", 'season', 'holiday', 'atemp' ,'windspeed', 'casual', 'registered', 'count'],axis=1)
print(dataset.dtypes)