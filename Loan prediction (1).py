#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("C:\\Users\\acer\\Downloads\\archive (3)\\train.csv")


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


pd.crosstab(df['Credit_History'], df['Loan_Status'], margins = True)


# In[9]:


pd.crosstab(df['Credit_History'], df['Loan_Status'])


# In[10]:


df.boxplot(column = 'ApplicantIncome')


# In[11]:


df['ApplicantIncome'].hist()


# In[12]:


df['ApplicantIncome'].hist(bins = 20)


# In[13]:


df['CoapplicantIncome'].hist(bins = 20)


# In[14]:


df.boxplot(column = 'ApplicantIncome', by= 'Education')


# In[15]:


df.boxplot(column = 'LoanAmount')


# In[16]:


df['LoanAmount'].hist(bins = 20)


# In[17]:


df['LoanAmount_log']=np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins = 20)


# In[18]:


df.isnull().sum()


# In[19]:


# filling the missing values


# In[20]:


df['Gender'].fillna(df['Gender'].mode()[0],inplace = True)


# In[21]:


df['Married'].fillna(df['Married'].mode()[0],inplace = True)


# In[22]:


df['Dependents'].fillna(df['Dependents'].mode()[0],inplace = True)


# In[23]:


df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace = True)


# In[24]:


df.LoanAmount = df.LoanAmount.fillna(df.LoanAmount.mean())


# In[25]:


df.LoanAmount_log = df.LoanAmount_log.fillna(df.LoanAmount_log.mean())


# In[26]:


df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace = True)


# In[27]:


df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0],inplace = True)


# In[28]:


df.isnull().sum()


# In[29]:


df['TotalIncome']= df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])


# In[30]:


df['TotalIncome_log'].hist(bins= 20)


# In[31]:


df.head()


# In[32]:


# Now divide into dependent and independent variable


# In[33]:


X = df.iloc[:,np.r_[1:5,9:11,13:15]].values # INDEPENDENT VARIABLE


# In[34]:


X


# In[35]:


y = df.iloc[:,12].values #Dependent variable loan eligiblity status


# In[36]:


y


# In[37]:


# Lets split into train and test data set


# In[38]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size = 0.2, random_state = 0)


# In[39]:


print(X_train)


# In[40]:


# Label encoder to convert categorical text into numerical form


# In[41]:


from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()


# In[42]:


for i in range(0,5):
    X_train[:,i]= labelencoder_x.fit_transform(X_train[:,i])


# In[43]:


X_train[:,i] = labelencoder_x.fit_transform(X_train[:,7])


# In[44]:


X_train


# In[45]:


labelencoder_y = LabelEncoder()
y_train = labelencoder_y.fit_transform(y_train)


# In[46]:


y_train


# In[47]:


for i in range(0,5):
    X_test[:,i]= labelencoder_x.fit_transform(X_test[:,i])


# In[48]:


X_test[:,i] = labelencoder_x.fit_transform(X_test[:,7])


# In[49]:


y_test = labelencoder_y.fit_transform(y_test)


# In[50]:


X_test


# In[51]:


y_test


# In[52]:


# scale the data


# In[53]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()


# In[54]:


X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)


# In[55]:


# completed all the preprocessing, handling missing values, handlind outlayers, scaling the data set, convert into numerical form


# In[56]:


# Applying algorithm
# uSING D cLASSIFIER


# In[57]:


from sklearn.tree import DecisionTreeClassifier
DTClassifier = DecisionTreeClassifier (criterion= 'entropy', random_state = 0)
DTClassifier.fit(X_train, y_train)


# In[58]:


y_pred = DTClassifier.predict(X_test)
y_pred


# In[59]:


from sklearn import metrics
print("The accuracy of decision trees is ", metrics.accuracy_score(y_pred,y_test))


# In[60]:


#APPLY ANOTHER ALOGORITHM


# In[61]:


from sklearn.naive_bayes import GaussianNB
NBClassifier = GaussianNB()
NBClassifier.fit(X_train, y_train)


# In[62]:


y_pred = NBClassifier.predict(X_test)


# In[63]:


y_pred


# In[64]:


from sklearn import metrics
print("The accuracy of decision trees is ", metrics.accuracy_score(y_pred,y_test))


# In[65]:


testdata = pd.read_csv("C:\\Users\\acer\\Downloads\\archive (3)\\train.csv")


# In[66]:


df.head()


# In[67]:


testdata.info()


# In[68]:


testdata.isnull().sum()


# In[69]:


testdata['Gender'].fillna(testdata['Gender'].mode()[0], inplace = True)


# In[70]:


testdata['Married'].fillna(testdata['Married'].mode()[0], inplace = True)
testdata['Dependents'].fillna(testdata['Dependents'].mode()[0], inplace = True)
testdata['Self_Employed'].fillna(testdata['Self_Employed'].mode()[0], inplace = True)
testdata['Self_Employed'].fillna(testdata['Self_Employed'].mode()[0], inplace = True)
testdata['Loan_Amount_Term'].fillna(testdata['Loan_Amount_Term'].mode()[0], inplace = True)
testdata['Credit_History'].fillna(testdata['Credit_History'].mode()[0], inplace = True)


# In[71]:


testdata.isnull().sum()


# In[72]:


testdata.boxplot(column = 'LoanAmount')


# In[73]:


testdata.boxplot(column = "ApplicantIncome")


# In[74]:


testdata.LoanAmount = testdata.LoanAmount.fillna(testdata.LoanAmount.mean())


# In[75]:


testdata.isnull().sum()


# In[76]:


testdata['LoanAmount'].hist()


# In[77]:


testdata['LoanAmount_log']= np.log(testdata['LoanAmount'])


# In[78]:


testdata['LoanAmount_log'].hist()


# In[79]:


testdata['TotalIncome'] = testdata['ApplicantIncome'] + testdata['CoapplicantIncome']


# In[80]:


testdata['Total_income_log'] = np.log(testdata['TotalIncome'])


# In[81]:


testdata.head()


# In[82]:


test = testdata.iloc[:,np.r_[1:5,9:11,13:15]].values


# In[83]:


# convert categorical to numerical data


# In[84]:


for i in range(0,5):
    test[:,i] = labelencoder_x.fit_transform(test[:,i])


# In[85]:


test[:,7]=labelencoder_x.fit_transform(test[:,7])


# In[86]:


test


# In[87]:


# scale the data


# In[88]:


test =ss.fit_transform(test)


# In[89]:


test


# In[90]:


pred = NBClassifier.predict(test)


# In[91]:


pred


# In[92]:


from sklearn.ensemble import RandomForestClassifier


# In[93]:


#fit the model on train data 
RF=RandomForestClassifier().fit(X_train,y_train)


# In[94]:


#predict on train 
train_preds4 = RF.predict(X_train)


# In[95]:


#accuracy on train
print("Model accuracy on train is: ", metrics.accuracy_score(y_train, train_preds4))


# In[102]:


metrics.accuracy_score(y_pred, y_test)


# In[110]:


from sklearn.linear_model import LogisticRegression


# In[112]:


logisticRegr = LogisticRegression(solver = 'lbfgs')


# In[113]:


logisticRegr.fit(X_train,y_train)


# In[114]:


train_preds4 = logisticRegr.predict(X_train)


# In[115]:


#accuracy on train
print("Model accuracy on train is: ", metrics.accuracy_score(y_train, train_preds4))


# In[ ]:




