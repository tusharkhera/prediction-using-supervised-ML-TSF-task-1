#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# data link "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"

# In[2]:


df = pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# ## Data Analysis

# In[6]:


df.plot()


# In[7]:


df.plot(kind='scatter', x='Hours', y='Scores', label='Scores')


# ## Splitting

# In[8]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
train_X = train_set.drop('Scores', axis=1)
train_y = train_set['Scores'].copy()
test_X = test_set.drop('Scores', axis=1)
test_y = test_set['Scores'].copy()


# In[9]:


print(len(train_X), len(train_y), len(test_X), len(test_y))


# ## Modelling

# In[10]:


from sklearn.ensemble import RandomForestRegressor
predict = RandomForestRegressor().fit(train_X, train_y)


# # prediction

# In[11]:


predicted_y = predict.predict(test_X)
df1 = pd.DataFrame({'Actual':test_y, 'Prediction':predicted_y})
df1


# # Evaluation

# In[12]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(test_y, predicted_y)
rmse = np.sqrt(mse)
print(f'rmse:{rmse}, mse:{mse}')


# # Task prediction 

# In[13]:


new_pred = predict.predict([[9.25]])
print(f'no. of hours={9.25}, percentage={new_pred}')


# # end of code
