# !/usr/bin/env python
# coding: utf-8

# # Byte Karma

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from faker import Faker
import random
import datetime
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

fake = Faker()

data = pd.DataFrame(columns=['CaseType', 'FilingDate', 'State', 'Priority'])


for _ in range(1000):
    case_type = random.choice(["Public Health Cases", "Medical Diagnosis Cases", "Environmental Pollution Cases", "Environmental Cases", "Missing Person Cases", "Unsolved Crimes Cases", "Child Protective Services Cases", "Elder Abuse Cases", "Criminal Cases", "Tax Cases", "Bankruptcy Cases", "Civil Cases", "Intellectual Property Cases", "Business Dispute Cases", "Employment Cases", "Family Law Cases", "Landlord-Tenant Cases", "Car Accident Cases", "Slip and Fall Cases", "Student Disciplinary Cases", "Special Education Cases", "International Law Cases", "Administrative Cases", "Immigration Cases"])
    filing_date = fake.date_between(start_date='-3y', end_date='today')
    State = random.choice(["Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal"])
    priority_mapping = {
        "Public Health Cases": 1,
        "Medical Diagnosis Cases": 2,
        "Environmental Pollution Cases": 3,
        "Environmental Cases": 4,
        "Missing Person Cases": 5,
        "Unsolved Crimes Cases": 6,
        "Child Protective Services Cases": 7,
        "Elder Abuse Cases": 8,
        "Criminal Cases": 9,
        "Tax Cases": 10,
        "Bankruptcy Cases": 11,
        "Civil Cases": 12,
        "Intellectual Property Cases": 13,
        "Business Dispute Cases": 14,
        "Employment Cases": 15,
        "Family Law Cases": 16,
        "Landlord-Tenant Cases": 17,
        "Car Accident Cases": 18,
        "Slip and Fall Cases": 10,
        "Student Disciplinary Cases": 20,
        "Special Education Cases": 21,
        "International Law Cases": 22,
        "Administrative Cases": 23,
        "Immigration Cases": 24,
    }
    priority = priority_mapping[case_type]

    data = data.append({'CaseType': case_type, 'FilingDate': filing_date, 'State': State, 'Priority': priority}, ignore_index=True)

data.to_csv('Cases.csv')


# In[2]:


data


# In[3]:


pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)


# In[4]:


data


# In[5]:


sns.countplot(x='CaseType',data=data)
plt.grid()
plt.xticks(rotation=90)
plt.show()


# In[6]:


for col in data.select_dtypes(include='object'):
    data[col]=pd.Categorical(data[col]).codes


# In[7]:


data


# In[8]:


x=data.drop('Priority',axis=1)
y=data['Priority']


# In[9]:


from sklearn.ensemble import RandomForestClassifier


# In[10]:


classifier=RandomForestClassifier(n_estimators=10)


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)


# In[13]:


classifier.fit(x_train,y_train)


# In[14]:


y_pred=classifier.predict(x_test)


# In[15]:


classifier.score(x_test,y_test)*100


# In[18]:


CaseType=int(input("Enter the case type(codes between 0 to 23):"))
FillingDate=int(input("Enter the filling date(codes between 0 to 667):"))
State=int(input("Enter the state(codes between 0 to 27):"))
list=[CaseType,FillingDate,State]
newdf=pd.DataFrame([list])
y_p=classifier.predict(newdf)
if(y_p<=8):
    print("Priority of the case is : High")
elif (y_p>8 and y_p<=16):
    print("Priority of the case is : Medium")
    
else:
    print("Priority of the case is : Low")


# In[ ]:




