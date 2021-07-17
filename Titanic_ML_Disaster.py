#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


# In[2]:


df1 = pd.read_csv('train.csv')


# In[3]:


df1.head()


# In[4]:


df1.isnull().sum()


# In[5]:


df1.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis=1,inplace=True)


# In[6]:


df1.head()


# In[7]:


df1.isnull().sum()


# In[8]:


df1['Age'].describe()


# In[9]:


df1['Age'].fillna(df1['Age'].mean(),inplace = True)


# In[10]:


df1.isnull().sum()


# In[11]:


l_sex_dummies=pd.get_dummies(df1['Sex'],drop_first=True)


# In[12]:


df1=pd.concat([df1,l_sex_dummies],axis=1)


# In[13]:


df1.head()


# In[14]:


df1.drop(['Sex'],axis=1,inplace=True)


# In[15]:


df1.head()


# In[16]:


from sklearn.preprocessing import StandardScaler
sts=StandardScaler()


# In[17]:


feature_scale = ['Age','Fare']
df1[feature_scale] = sts.fit_transform(df1[feature_scale])


# In[18]:


df1.head()


# In[ ]:





# In[19]:


x= df1.drop(['Survived'],axis=1)
y= df1['Survived']


# In[20]:


x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2)


# In[21]:


model_svc=SVC(C=100,kernel='rbf')


# In[22]:


model_svc.fit(x_train,y_train)


# In[23]:


prediction = model_svc.predict(x_test)


# In[24]:


from sklearn.metrics import classification_report


# In[25]:


print(classification_report(y_test,prediction))


# In[26]:


model_param = {
    'DecisionTreeClassifier':{
        'model':DecisionTreeClassifier(),
        'param':{
            'criterion':['gini','entropy']
        }
    },
        'KNeighborsClassifier':{
            'model':KNeighborsClassifier(),
            'param':{
                'n_neighbors':[5,10,15,20,25]
        }
            
    },
        'SVC':{
            'model':SVC(),
            'param':{
                'kernel':['rbf','linear','sigmoid'],
                'C':[0.1,1,10,100]
            }
            
        }   
}


# In[27]:


scores=[]
for model_name, mp in model_param.items():
    model_selection = GridSearchCV(estimator = mp['model'],param_grid=mp['param'],cv=5,return_train_score=False)
    model_selection.fit(x,y)
    scores.append({'model':model_name,
                  'best_score':model_selection.best_score_,
                  'best_params':model_selection.best_params_})


# In[28]:


df_model_score=pd.DataFrame(scores,columns=['model','best_score','best_params'])
df_model_score


# In[29]:


model_svc=SVC(C=100,kernel='rbf')


# In[30]:


model_svc.fit(x,y)


# In[31]:


df2=pd.read_csv('test.csv')


# In[32]:


df2.head()


# In[33]:


df3=df2.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis=1)


# In[34]:


df3.isnull().sum()


# In[35]:


df3['Age'].fillna(df3['Age'].mean(),inplace =True)
df3['Fare'].fillna(df3['Fare'].mean(),inplace=True)


# In[36]:


l_sex_dummies=pd.get_dummies(df3['Sex'],drop_first=True)
df3=pd.concat([df3,l_sex_dummies],axis=1)
df3.drop(['Sex'],axis=1,inplace=True)


# In[37]:


df3.head()


# In[38]:


df3[feature_scale] = sts.fit_transform(df3[feature_scale])


# In[39]:


df3.head()


# In[40]:


y_predicted = model_svc.predict(df3)


# In[41]:


result = pd.DataFrame({
    "PassengerId":df2['PassengerId'],
    "Survived": y_predicted
})


# In[42]:


print(result)


# In[43]:


result.to_csv('Titanic_ML_Disaster_result.csv',index=False)

