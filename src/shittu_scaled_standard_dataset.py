#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[22]:


df = pd.read_csv("../data/scaled_standard_dataset.csv")


# In[23]:


df.head()


# In[24]:


correl =df.corr(method='pearson').round(2)
plt.figure(figsize=(15,15))
sns.heatmap(correl,annot=True)
plt.show()


# In[25]:


correl =df.corr(method='spearman').round(2)
plt.figure(figsize=(15,15))
sns.heatmap(correl,annot=True)
plt.show()


# In[26]:


correl =df.corr(method='kendall').round(2)
plt.figure(figsize=(15,15))
sns.heatmap(correl,annot=True)
plt.show()


# In[29]:


# Selecting only the numeric columns for correlation calculation
numeric_columns = df.select_dtypes(include=['int64', 'float64'])


# In[31]:


correlation_matrix = numeric_columns.corr()


# In[32]:


correlation_matrix


# In[ ]:




