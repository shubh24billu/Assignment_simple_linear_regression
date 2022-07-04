#!/usr/bin/env python
# coding: utf-8

# ## Build a simple linear regression model by performing EDA and do necessary transformations and select the best model using R or Python.
# 

# ### Predict delivery time using sorting time 

# In[1]:


# importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, LassoCV, Lasso, RidgeCV, Ridge, ElasticNetCV, ElasticNet
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")


# In[2]:


#loading Dataset
deliv= pd.read_csv("delivery_time.csv")
deliv.head()


# In[3]:


#renaming columns
deliv=deliv.rename(columns={'Delivery Time':'Delivery_time'})
deliv=deliv.rename(columns={"Sorting Time":"Sorting_time"})


# In[4]:


# fetching information by using info and describe function to understand the data well.
deliv.describe(),deliv.info()


# In[5]:


#plotting scatter plot to understand relationship between the target and the independent variable.
plt.scatter(deliv['Delivery_time'],deliv['Sorting_time']);


# In[6]:


sns.histplot(deliv['Delivery_time']);


# In[7]:


deliv.corr()


# In[8]:


#defining x and y variable
x=np.array([deliv["Sorting_time"]]).transpose()
y=np.array(deliv["Delivery_time"])
x.shape,y.shape


# In[9]:


#training model by using ols and sklearn 
model_ols = sm.OLS(y,sm.add_constant(x)).fit()
model_sk = LinearRegression()
model_sk.fit(x,y)


# In[10]:


model_sk.coef_,model_sk.score(x,y),model_ols.rsquared,model_ols.summary()


# In[11]:


prediction = model_sk.predict(x)
mse = mean_squared_error(y,prediction)
rmse = np.sqrt(mse)
mse,rmse


# In[12]:


#cooks distance for influence point analysis
model_influence = model_ols.get_influence()
(c, _) = model_influence.cooks_distance


# In[13]:


fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(x)), np.round(c, 3))
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()


# In[14]:


(np.argmax(c),np.max(c[c>0.2]))


# ### Here some points which are having c>0.2 are influential points so we will try to remove those and see the score.

# In[15]:


#removing the 4th row 
deliv_new= pd.read_csv("delivery_time.csv")
deliv_new1 = deliv_new.drop(deliv_new.index[[4,8]],axis=0).reset_index()
deliv_new1 = deliv_new1[['Delivery Time','Sorting Time']]
deliv_new1.rename(columns={'Delivery Time':'Delivery_time',"Sorting Time":"Sorting_time"}, inplace=True)


# In[16]:


deliv_new1.info()


# In[17]:


x1=np.array([deliv_new1["Sorting_time"]]).transpose()
y1=np.array(deliv_new1["Delivery_time"])
x1.shape,y1.shape


# In[18]:


model_ols1 = sm.OLS(y1,sm.add_constant(x1)).fit()
model_sk1 = LinearRegression()
model_sk1.fit(x1,y1)


# In[19]:


model_sk1.coef_,model_sk1.score(x,y),model_ols1.rsquared,model_ols1.summary()


# ### The score is definitely increasing for the model by removing outliers and also reduced the rmse 

# In[20]:


prediction1 = model_sk1.predict(x1)
mse1 = mean_squared_error(y1,prediction1)
rmse1 = np.sqrt(mse1)
mse1,rmse1


# In[21]:


model_influence1 = model_ols1.get_influence()
(c, _) = model_influence1.cooks_distance


# In[22]:


fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(x1)), np.round(c, 3))
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()


# In[23]:


4/19


# In[24]:


deliv_new2 = deliv_new1.drop(deliv_new1.index[[18]],axis=0).reset_index()
deliv_new2 = deliv_new2[["Delivery_time","Sorting_time"]]
deliv_new2.info()


# In[25]:


x2=np.array([deliv_new2["Sorting_time"]]).transpose()
y2=np.array(deliv_new2["Delivery_time"])
x2.shape,y2.shape


# In[26]:


model_ols2 = sm.OLS(y2,sm.add_constant(x2)).fit()
model_sk2 = LinearRegression()
model_sk2.fit(x2,y2)


# In[27]:


model_sk2.coef_,model_sk2.score(x,y),model_ols2.rsquared,model_ols2.summary()


# ## After removing the outliers this is the final accuracy we are getting for the improved model

# # The accuracy of the final model is 83.3%

# In[ ]:




