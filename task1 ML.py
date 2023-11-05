#!/usr/bin/env python
# coding: utf-8

# In[1]:


# TASK 1


# In[2]:



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
     


# In[2]:


import pandas as pd


# In[3]:


file = pd.read_csv('C:/Users/ASUS/OneDrive/Documents/excel/train.csv')
file.head()


# In[4]:


file.tail()


# In[5]:


file.shape


# In[6]:


file.info()


# In[9]:


file1 = file[['LotArea', 'FullBath', 'HalfBath', 'BedroomAbvGr','SalePrice']]
file1.head()


# In[11]:


file1.duplicated().sum()


# In[12]:


file1.drop_duplicates()


# In[14]:


import matplotlib.pyplot as plt


# In[18]:


import numpy as np
import seaborn as sns


# In[19]:


# correlation matrix
fig, axs = plt.subplots(figsize=(8, 4))
mat = file1.corr(method = 'pearson')
mask = np.triu(np.ones_like(mat, dtype=bool))
sns.heatmap(mat, mask=mask, cmap = sns.color_palette('Blues'), vmax=1, center=0, annot = False, linewidths=.5, cbar_kws={'shrink': .7})
axs.set_title("Correlation Map")
plt.show()


# In[21]:


fig, axs = plt.subplots(figsize = (16,6))

axs = plt.scatter(x=file1.LotArea, y=file.SalePrice)
plt.ylabel('Sale Price')
plt.xlabel('Lot size in square feet')
plt.title('Scatter Relating SalePrice to LotArea')


# In[22]:


def detectOutliers():
    fig, axs = plt.subplots(2,3, figsize = (10,5))
    plt1 = sns.boxplot(file1['SalePrice'], ax = axs[0,0])
    plt2 = sns.boxplot(file1['LotArea'], ax = axs[0,1])
    plt3 = sns.boxplot(file1['FullBath'], ax = axs[0,2])
    plt1 = sns.boxplot(file1['HalfBath'], ax = axs[1,0])
    plt2 = sns.boxplot(file1['BedroomAbvGr'], ax = axs[1,1])
    plt.tight_layout()
detectOutliers()


# In[24]:


# Outlier reduction for price
Q1 = file1.SalePrice.quantile(0.25)
Q3 = file1.SalePrice.quantile(0.75)
IQR = Q3 - Q1
file1 = file1[(file1.SalePrice >= Q1 - 1.5*IQR) & (file1.SalePrice <= Q3 + 1.5*IQR)]
 
# Outlier reduction for area
Q1 = file1.LotArea.quantile(0.25)
Q3 = file1.LotArea.quantile(0.75)
IQR = Q3 - Q1
file1 = file1[(file1.LotArea >= Q1 - 1.5*IQR) & (file1.LotArea <= Q3 + 1.5*IQR)]

# Outlier reduction for area
Q1 = file1.BedroomAbvGr.quantile(0.25)
Q3 = file1.BedroomAbvGr.quantile(0.75)
IQR = Q3 - Q1
file1 = file1[(file1.BedroomAbvGr >= Q1 - 1.5*IQR) & (file1.BedroomAbvGr <= Q3 + 1.5*IQR)]


# In[25]:


detectOutliers()


# In[26]:


file1.dtypes


# In[27]:


X = file1.drop('SalePrice',axis=1)
y = file1['SalePrice']


# In[29]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=33)
     

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)


# In[31]:


#


# In[32]:


y_pred = lr_model.predict(X_test)


# In[34]:



from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
     


# In[35]:


r2 = round(r2_score(y_test,y_pred),5)
print('Coefficient of determination R2: ', r2)


# In[36]:


plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='-')
plt.xlabel('Actual Values')
plt.ylabel('Prediction')
plt.title('Scatter chart - Linear regression model')


# In[ ]:




