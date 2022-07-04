#!/usr/bin/env python
# coding: utf-8

# ## Build a simple linear regression model by performing EDA and do necessary transformations and select the best model using R or Python.

# ### Build a prediction model for Salary_hike

# In[1]:


# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")


# In[2]:


#loading Dataset
sal=pd.read_csv("Salary_Data.csv")
sal.head()


# In[3]:


# fetching information by using info and describe function to understand the data well.
sal.describe(),sal.info()


# In[4]:


#plotting scatter plot to understand relationship between the target and the independent variable.
plt.scatter(sal['Salary'],sal['YearsExperience']);


# In[5]:


sal.corr()


# In[6]:


#defining x and y variable
x=np.array([sal['YearsExperience']]).transpose()
y=sal['Salary']


# In[7]:


#training model by using ols and sklearn 
model_ols = sm.OLS(y,sm.add_constant(x)).fit()
model_sk = LinearRegression()
model_sk.fit(x,y)


# In[8]:


model_sk.coef_,model_sk.score(x,y),model_ols.rsquared,model_ols.summary()


# In[9]:


prediction = model_sk.predict(x)
mse = mean_squared_error(y,prediction)
rmse = np.sqrt(mse)
mse,rmse


# In[10]:


model_sk.intercept_,model_sk.score(x,y)


# cooks distance analysis for influential points.
# here 2 points are influential so we can remove it but our score is pretty good so i am not acting on it for now

# In[11]:


#cooks distance for influence point analysis
model_ols_influence = model_ols.get_influence()
(c, _) = model_ols_influence.cooks_distance


# In[12]:


fig= plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(x)),np.round(c,3));


# In[13]:


N = len(sal)
4/N


# ### There are no such influential point or outlier in the data so this is our final accuracy

# In[14]:


qqplot=sm.qqplot(model_ols.resid,line='q') # line = 45 to draw the diagnoal line
plt.title("Normal Q-Q plot of residuals")
plt.show()


# # The accuracy of the final model is 95.7%
# 

# In[ ]:




