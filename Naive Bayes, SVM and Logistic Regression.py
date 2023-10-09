#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("pd_speech_features.csv")


# In[3]:


df


# In[4]:


inputs = df.drop("class",axis = "columns")


# In[5]:


inputs.head()


# In[6]:


target = df["class"]
target.head()


# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputs,target,test_size=0.2)


# In[30]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=20)


# In[31]:


model.fit(X_train,y_train)


# In[32]:


v = model.score(X_test, y_test)


# In[33]:


print(v*100)


# In[18]:


from sklearn.metrics import classification_report


# ## Naive Bayes

# In[9]:


from sklearn.naive_bayes import GaussianNB


# In[13]:


nb_model = GaussianNB()
nb_model = nb_model.fit(X_train, y_train)


# In[16]:


y_pred = nb_model.predict(X_test)


# In[19]:


print(classification_report(y_test,y_pred))


# In[29]:


ck = nb_model.score(X_test, y_test)
print(ck)


# ## Logistic Regression

# In[20]:


from sklearn.linear_model import LogisticRegression


# In[21]:


lg_model = LogisticRegression()
lg_model = lg_model.fit(X_train, y_train)


# In[22]:


y_pred = lg_model.predict(X_test)


# In[23]:


print(classification_report(y_test,y_pred))


# In[30]:


ch = lg_model.score(X_test, y_test)
print(ch)


# In[ ]:





# ## SVM

# In[24]:


from sklearn.svm import SVC


# In[25]:


svm_model = SVC()
svm_model = svm_model.fit(X_train, y_train)


# In[26]:


y_pred = svm_model.predict(X_test)


# In[27]:


print(classification_report(y_test,y_pred))


# In[ ]:




