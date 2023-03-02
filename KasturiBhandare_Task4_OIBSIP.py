#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


# # Preprocessing of the data

# In[3]:


raw_mail = pd.read_csv(r'C:\Users\Kasturi\Desktop\Oasis\spam.csv', encoding = 'latin1')
raw_mail
mail_data = raw_mail.where((pd.notnull(raw_mail)), '')


# In[4]:


mail_data.shape


# In[6]:


mail_data.head()


# In[9]:


# label the spam emails as 0 and the non spam emails as 1
mail_data.loc[mail_data['v1'] == 'spam', 'v1',] = 0
mail_data.loc[mail_data['v1'] == 'ham', 'v1',] = 1


# In[10]:


# separating the data as text and labelling X --> text; Y --> label
X = mail_data['v2']
Y = mail_data['v1']


# In[13]:


print(X)


# In[14]:


print(Y)


# In[15]:


# split the data as train data and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=3)


# # Feature extraction

# In[16]:


#transform the text data to feature vectors that can be used as input to the svm model using TfidVectorizer
#covert the text to lower case letter

feature_extraction = TfidfVectorizer(min_df = 1, stop_words = 'english', lowercase = True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)


# In[17]:


# convert the Y_train and Y_test values to integers

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


# # Training the model using Support Vector Machine(SVM)

# In[18]:


# training the support vector machine model with training data 

model = LinearSVC()
model.fit(X_train_features, Y_train)


# # Evaluating the model

# In[19]:


#prediction on training data
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)


# In[20]:


print('Accuracy on training data : ',accuracy_on_training_data)


# In[34]:


#prediction on test data 
prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print('Accuracy: ',accuracy_on_test_data)
print('Accuracy of the model is: {:.3f}'.format(accuracy_on_test_data*100),'%')


# # Prediction using an email

# In[22]:


input_mail = ["Even my brother is not like to speak with me. They treat me like aids patent.,,,"]

#convert text to feature 

input_mail_features = feature_extraction.transform(input_mail)


# In[23]:


#making predictions

prediction = model.predict(input_mail_features)


# In[35]:


#Spam email means 0, ham email means 1

print(prediction)

if (prediction[0] == 1):
              print("IT IS NOT A SPAM MAIL")
else :
              print("IT IS A SPAM MAIL")


# In[ ]:




