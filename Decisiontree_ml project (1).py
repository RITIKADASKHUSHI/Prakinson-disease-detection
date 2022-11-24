#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv("pd_speech_features.csv")
df


# In[2]:


inputs = df.drop('class',axis='columns')


# In[3]:


target = df['class']


# In[10]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X_train,X_test,y_train,y_test=train_test_split(inputs,target,test_size=0.2)


# In[11]:


from sklearn import tree
model = tree.DecisionTreeClassifier()


# In[12]:


model.fit(X_train,y_train)


# In[13]:


pred=model.predict(X_test)


# In[14]:


sc=accuracy_score(y_test,pred)


# In[15]:


print(sc*100)


# In[ ]:




