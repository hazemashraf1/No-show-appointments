#!/usr/bin/env python
# coding: utf-8

# # Project: Investigate a Dataset (No-show appointments)
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# This dataset collects information from 100k medical appointments in Brazil and is focused on the question of whether or not patients show up for their appointment
# 

# <a id='wrangling'></a>
# ## Data Wrangling

# In[137]:


import pandas as pd
import numpy as np
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
 


# In[138]:


# load data from given dataset
df=pd.read_csv(r"C:\Users\hazem\Desktop\noshowappointments-kagglev2-may-2016.csv")
# know the shape
df.shape
 


# In[139]:


#lets see our data
df.head()


# In[140]:


#whats the current columns names
df.columns


# In[141]:


#fixing some names
df.columns=['patient_id','appointment_id','gender','scheduled_day','appointment_day','age','neighboourhood','scholarship','hypertension','diabetes','alcoholism','handcap','sms_recieved','no_show']
df.columns


# In[142]:


#check after changing 
df.head()


# In[143]:


#check the info of the dataset
df.info()


# In[144]:


#lets see how many missed there scheduled appointments ---conc(1)
df['no_show'].value_counts()


# In[145]:


#lets see how many missed there scheduled appointments ---conc(1)
missed=len(df.query('no_show == "Yes"'))
total=df.shape[0]
ratio=int(missed/total*100)
ratio


# In[146]:


# change patient_id type from float to int as it should--- no missing values
df['patient_id']=df['patient_id'].astype('int64')
df.info()


# In[147]:


#check data again
df.head()


# In[148]:


# change type of scheduled_day , appointment_day  to datetime to be readable
df['scheduled_day']=pd.to_datetime(df['scheduled_day']).dt.date.astype('datetime64')
df['appointment_day']=pd.to_datetime(df['appointment_day']).dt.date.astype('datetime64')

df.head()


# <a id='eda'></a>
# ## Exploratory Data Analysis

# In[149]:


df.describe()


# In[150]:


# check for duplicates as it seems logic to see duplicates which means same patients visiting the hospital
df['patient_id'].duplicated().sum()


# In[151]:


# first and last scheduled visit --- conc(2)
first_scheduled =df['scheduled_day'].min()
last_scheduled=df['scheduled_day'].max()

# first and last rescheduled appointment
first_appointment=df['appointment_day'].min()
last_appointment=df['appointment_day'].max()

print('First Scheduled visit: {}'.format(first_scheduled))
print('Last Scheduled visit: {}'.format(last_scheduled))
print('First reScheduled visit: {}'.format(first_appointment))
print('Last reScheduled visit: {}'.format(last_appointment))


# In[152]:


#know the week day of scheduled visit 
df['scheduled_weekday']= df.scheduled_day.dt.day_name()
df.head()


# In[153]:


# know the appointmnets distrubtion during the week --- conc(2)
df['scheduled_weekday'].value_counts()


# In[154]:


# how many days does the patient wait after change the scheduled_day to real appointment_day----- conc(3)
df['rescheduling_duration']=(df.appointment_day - df.scheduled_day).dt.days
df['rescheduling_duration'].describe()


# In[155]:


# how many rescheduled for the same day----- conc(3)
same_day= df[(df.rescheduling_duration== 0)].rescheduling_duration.value_counts()
same_day


# In[156]:


# how many rescheduled for the same day and also missed the visit----- conc(3)
missed_same_day=len(df.query('rescheduling_duration== 0 and no_show== "Yes"'))
missed_same_day


# In[157]:


df.describe()


# - Age: 25% of patients are years old ,50% of patients are 37 years old , 75% of patients are 55 years old ,and the oldest patient is 115 years old
# 
# - Rescheduling patients wait a duration on average 10 days , 25% of patients wait 0 day ,50%  of patients wait 4
#   75% of patients wait 15 days , and max waiting duration reached 179 days
# -  scholarship	
# - hypertension	: 75% of patients dont have hypertension
# - diabetes      : 75% of patients dont have diabetes
# - alcoholism    : 75% of patients dont have alcoholism
# - handcap       : Different from the other categories its have 4 classes and 75% of patients dont have handicap
# - sms recieved  : 75% of patients have recieved sms regarding appointments
# 

# # will see furhther visualization tools and investigation below
# ### Research Question 1 ( how many missed there scheduled appointments)

# In[158]:


#lets see how many missed there scheduled appointments ---conc(1)*

noshow_count = sns.countplot(x=df.no_show, data=df) 
noshow_count.set_title("Show VS. No Show")
plt.show();
pd.DataFrame(df.groupby(['no_show'])[['patient_id']].count())


# In[159]:


#distribution of week day appointmnents ----- conc(5)
bar_chart_weekday= sns.countplot(x=df.scheduled_weekday, data=df)
bar_chart_weekday.set_title("scheduled_weekday")
plt.show();
pd.DataFrame(df.groupby(['scheduled_weekday'])[['patient_id']].count())


# he visits along the week nearly equal with highest number of visits on Tuesday with 26168 visit and lowest visits number on   Saturday with 24 visits.

# In[160]:


#age distribution in data set----- conc(6)
df['age'].describe()


# In[161]:


#further invistgation in age distribution ----- conc(6)
plt.figure(figsize=(14,4))
plt.xticks(rotation=0)
age_boxplot= sns.boxplot(x=df.age)


# 25%  of patients are 18 years old
# 50%  of patients are 37 years old
# 75%  of patients are 55 years old
# and we have an outliner with a max age 115 years old

# ### Research Question 2  ( The Relation between Age ,No Show and  The Appointments number)

# In[162]:


#lets see the relation between the age and the appointments number ----- conc(7)
plt.figure(figsize=(20,8))
plt.xticks(rotation=90)
appointments_per_age= sns.countplot(x=df.age)
appointments_per_age.set_title("appointments per age")
plt.show()
 


# we have a peak at 0 which indicates there is alot of infants (newborn) who have appointments ,compared to the rest age             distribution ,rest of the patients age seems nearly equally distributed and start to decrease from 59 years old

# In[163]:


#relation between age and no show
df.boxplot(column=['age'],by=['no_show'],rot=0)
plt.ylabel('age')
pd.DataFrame(df.groupby(['no_show'])['age'].describe().loc[:,['mean','std']])


# In[164]:


#quick gender analysis ---- conc(8)
pd.DataFrame(df.groupby(['gender'])[['patient_id']].count())


# In[165]:


#gender count--- conc(8) 
df.groupby(['gender'])['patient_id'].count().plot(kind='bar').set_ylabel('count')
pd.DataFrame(df.groupby(['gender'])[['patient_id']].count())


# ### Research Question 3  ( The Relation between gender and no show)

# In[166]:


#lets invistegate relation between gender and no show--- conc(8)
#first get the count of each gender
total_gender=df.shape[0]
total_male= len(df.loc[df['gender'] == "M"])
total_female= len(df.loc[df['gender'] == "F"])
percentage_male=int(round(total_male/total_gender * 100))
percentage_female=int(round(total_female/total_gender * 100))

#second we need who missed there appointments
missed_male=len(df.query('gender == "M" and no_show== "Yes"'))
missed_female=len(df.query('gender == "F" and no_show== "Yes"'))

#finally calculate the precentage of each gender
ratio_male=int(round(missed_male/total_male * 100))
ratio_female=ratio_male=int(round(missed_female/total_female * 100))
print("There is {}% Males and {}% Females".format(percentage_male,percentage_female))
 
print("There is {}% Males missed there appointments from total Males of : {}".format(ratio_male,total_male))
print("There is {}% Females missed there appointments from total Females of : {}".format(ratio_female,total_female))


# In[167]:


#relation between gender and no show--- conc(8)
ax = sns.countplot(x=df.gender, hue=df.no_show, data=df)
ax.set_title("gender relation to no show")
x_ticks_labels=['Female', 'Male']
plt.show();
pd.DataFrame(df.groupby(['gender'])[['no_show']].count())


# ### Research Question 4  (  Whats the range of most of the schedules)

# In[168]:


# whats the range of most of the schedules---- conc(9)
fig = plt.figure(figsize=(16, 8))
schedule_range = fig.add_subplot(1, 1, 1)
schedule_range.set_xlabel('scheduled_day')
df['scheduled_day'].hist();


# Scheduled Appointments histrogram shows that : its left skewed ,which means scheduled appointment mostly made between march-   2016 and june-2016

# In[169]:


# Quick visualization for the raw data distribution -categorical type  ----- conc(10)
df.hist(figsize=(16,16));


# ### Research Question 5  (  which category affect show or no show to the appointmnet)

# ### Hypertension

# In[170]:


#investigate the no show ratio of hypertension
fig=plt.figure(figsize=(16,14))

graph= fig.add_subplot(3, 3, 1+i) 
df.groupby(['hypertension', 'no_show'])['hypertension'].count().unstack('no_show').plot(ax=graph, kind='bar', stacked=True)     
df.groupby(['hypertension', 'no_show'])['hypertension'].count().unstack('no_show') 


# In[171]:


#investigate the no show ratio of hypertension
df['no_show_%'] = np.where(df['no_show']=='Yes', 1, 0)
df[['hypertension', 'no_show_%']].groupby(['hypertension'], as_index=False).mean().sort_values(by='no_show_%', ascending=False)


# 17% of patients who have hypertension did not showed to the appointment which is lower than average with 3% of no show , which is indicates  more care to the shceduled appointment from the first time

# ### Diabetes

# In[172]:


#investigate the no show ratio of diabetes
fig=plt.figure(figsize=(16,14))

graph= fig.add_subplot(3, 3, 1+i) 
df.groupby(['diabetes', 'no_show'])['diabetes'].count().unstack('no_show').plot(ax=graph, kind='bar', stacked=True)     
df.groupby(['diabetes', 'no_show'])['diabetes'].count().unstack('no_show') 


# In[173]:


#investigate the no show ratio of diabetes
df[['diabetes', 'no_show_%']].groupby(['diabetes'], as_index=False).mean().sort_values(by='no_show_%', ascending=False)


# 18% of patients who have diabetes did not showed to the appointment which is lower than average with 2% of no show , which is indicates more care to the shceduled appointment from the first time

# ### Alcoholism

# In[174]:


#investigate the no show ratio of alcoholism

fig=plt.figure(figsize=(16,14))

graph= fig.add_subplot(3, 3, 1+i) 
df.groupby(['alcoholism', 'no_show'])['alcoholism'].count().unstack('no_show').plot(ax=graph, kind='bar', stacked=True)     
df.groupby(['alcoholism', 'no_show'])['alcoholism'].count().unstack('no_show') 


# In[175]:


#investigate the no show ratio of alcoholism
df[['alcoholism', 'no_show_%']].groupby(['alcoholism'], as_index=False).mean().sort_values(by='no_show_%', ascending=False)


# 20% of patients who have alcoholism did not showed to the appointment which is the average of no show , which is indicates same  care to the shceduled appointment as the not alcoholic

# ### Handcap

# In[176]:


#investigate the no show ratio of handcap

fig=plt.figure(figsize=(16,14))

graph= fig.add_subplot(3, 3, 1+i) 
df.groupby(['handcap', 'no_show'])['handcap'].count().unstack('no_show').plot(ax=graph, kind='bar', stacked=True)     
df.groupby(['handcap', 'no_show'])['handcap'].count().unstack('no_show')


# In[177]:


#investigate the no show ratio of handcap
df[['handcap', 'no_show_%']].groupby(['handcap'], as_index=False).mean().sort_values(by='no_show_%', ascending=False)


# ### SMS Recieved

# In[178]:


#investigate the no show ratio of sms_recieved

fig=plt.figure(figsize=(16,14))

graph= fig.add_subplot(3, 3, 1+i) 
df.groupby(['sms_recieved', 'no_show'])['sms_recieved'].count().unstack('no_show').plot(ax=graph, kind='bar', stacked=True)     
df.groupby(['sms_recieved', 'no_show'])['sms_recieved'].count().unstack('no_show')


# In[179]:


#investigate the no show ratio of sms_recieved
df[['sms_recieved', 'no_show_%']].groupby(['sms_recieved'], as_index=False).mean().sort_values(by='no_show_%', ascending=False)


# its surprising here to see 27% who have recieved sms didnt show up which above average with 7% , and only 16% who have not recieved sms did not show up which is below average with 4%

# In[180]:


#sumarize investigation to know which category affect show or no show to the appointmnet----- conc(10)
variables=['scholarship','hypertension','diabetes','alcoholism','handcap','sms_recieved','scheduled_weekday']

fig=plt.figure(figsize=(16,14))
for i , variable in enumerate (variables):
    graph= fig.add_subplot(3, 3, 1+i)
    df.groupby([variable, 'no_show'])[variable].count().unstack('no_show').plot(ax=graph, kind='bar', stacked=True)
    


# # Conclusions

# - First i have discovered the data set and manage to rename some columns ,clean some data, add new columns which will help to     understand the dataset more, wrote some notes along the Exploratory Data Analysis , but as we go further we can see that we     need more information and data to get the prediction trend more accurate , we need to apply more advanced analysis.
# 
# After analysis we found the following:
# 
# - There are 88208 who showed for scheduled appointment and 22319 didnt show for there appointment which is 20% from the total appointments, there are 48228 patients who have been repeated in the dataset which indicates several visits for the same Patient.
# 
# - The visits along the week nearly equal with highest number of visits on Tuesday with 26168 visit and lowest visits number on   Saturday with 24 visits. 
# 
# - Found that First Scheduled visit on: 2015-11-10 ,Last Scheduled visit on: 2016-06-08.
# - Also found that First rescheduled visit on: 2016-04-29 ,Last rescheduled visit on: 2016-06-08.
# - Scheduled Appointments histrogram shows that : its left skewed ,which means scheduled appointment mostly made between march-   2016 and june-2016
# 
# - There are 38563 Patients who rescheduled for the same day,and 1792 Patients missed the rescheduled appointment for the same     day with 4.6%
# 
# - After rescheduling patients wait a duration on average 10 days , 25% of patients wait 0 day ,50%  of patients wait 4
#   75% of patients wait 15 days , and max waiting duration reached 179 days
#   
# - Age distribution shows that : 25% of patients are years old ,50% of patients are 37 years old , 75% of patients are             55 years old ,and the oldest patient is 115 years old
# 
# - Appointments per Age shows that : there is alot of infants (newborn) who have appointments compared to the rest age             distribution ,rest of the patients age seems nearly equally distributed and start to decrease from 59 years old
# 
# - Gender analysis shows that :Female count is 71840 with 65%  ,and Male count is 38687 with 35% 
# 
# - Appointments per Gender : 38687 Appointments by Males which  20% of them missed there scheduled Appointments 
#   and 71840 Appointments by Females which  20% of them missed there scheduled Appointments too
#   
# - 27% who have recieved sms didnt show up which above average with 7% , and only 16% who have not recieved sms did not show up which is below average with 4%
# 
# - Finally Categorical type of data as shown in the analysis such as: scholarship ,hypertension ,diabetes, alcoholism, handcap ,
#   sms recieved , scheduled weekday are nearly the same with average no show precentage 20 % indicates no certain prediction       trend or key to the no show.
# 
# 
# 
#         
#  
# 
# 
# 
# 
#  
# 
