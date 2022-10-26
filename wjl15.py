
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV


# In[3]:


import seaborn as sns; sns.set()
import matplotlib.pyplot as plt


# In[4]:


np.random.seed(42)


# # 1.white wine analysis

# ## data preprossing

# In[5]:


white_data = pd.read_csv('winequality-white.csv',sep = ';')


# In[6]:


print(white_data.info())
print(white_data.head())


# In[7]:


sns.distplot(white_data.quality)
plt.show()


# In[8]:


#normalize data
for col in white_data.columns[:-1]:
    white_data[col] = (white_data[col]-white_data[col].min())/(white_data[col].max()-white_data[col].min())


# In[9]:


#remove outliers
for col in white_data.columns[:-1]:
    white_data = white_data[white_data[col]<0.9]
len(white_data)


# In[10]:


X = white_data.iloc[:,:-1].values
y = white_data.iloc[:,-1].values


# ### metrics defination

# In[11]:


def accuracy(y_pred,y_true):
    ont = 0
    for i in range(len(y_pred)):
        if int(y_pred[i]) == y_true[i] or int(y_pred[i])+1 == y_true[i]:
            ont += 1
    return ont*1.00/len(y_true)


# In[12]:


def rmse(y_pred,y_true):
    return np.sqrt(np.mean(np.square(y_pred-y_true)))


# ## baseline

# In[13]:


def baseline(X,y):
    lr = LinearRegression()
    kf = KFold(n_splits=5,shuffle = True)
    acc = []
    rms = []
    for train_index,test_index in kf.split(X):
        x_train = X[train_index]
        y_train = y[train_index]
        x_test = X[test_index]
        y_test = y[test_index]
        lr.fit(x_train,y_train)
        y_pred = lr.predict(x_test)
        acc.append(accuracy(y_pred,y_test))
        rms.append(rmse(y_pred,y_test))
    print(np.mean(acc),np.mean(rms))


# In[14]:


#baseline cross validation
lr = LinearRegression()
kf = KFold(n_splits=5,shuffle = True)
acc = []
rms = []
for train_index,test_index in kf.split(X):
    x_train = X[train_index]
    y_train = y[train_index]
    x_test = X[test_index]
    y_test = y[test_index]
    lr.fit(x_train,y_train)
    y_pred = lr.predict(x_test)
    acc.append(accuracy(y_pred,y_test))
    rms.append(rmse(y_pred,y_test))
print(np.mean(acc),np.mean(rms))


# ## advanced algorithms

# In[15]:


#three methods results
def methods(feature,X,y):
    kr = KNeighborsRegressor(n_neighbors = 20)
    gbr = GradientBoostingRegressor(max_depth = 3,max_features = 'log2',loss = 'ls')
    svm = SVR(C= 3, epsilon= 0.1, gamma= 1)
    kf = KFold(n_splits=5,shuffle = True)
    for i in range(1):
        x = X[:,feature]
        acc_kr = []
        acc_gbr = []
        acc_svm = []
        rms_kr = []
        rms_gbr = []
        rms_svm = []
        for train_index,test_index in kf.split(x):
            x_train = x[train_index]
            y_train = y[train_index]
            x_test = x[test_index]
            y_test = y[test_index]
            svm.fit(x_train,y_train)
            y_pred = svm.predict(x_test)
            acc_svm.append(accuracy(y_pred,y_test))
            rms_svm.append(rmse(y_pred,y_test))
            kr.fit(x_train,y_train)
            y_pred = kr.predict(x_test)
            acc_kr.append(accuracy(y_pred,y_test))
            rms_kr.append(rmse(y_pred,y_test))
            gbr.fit(x_train,y_train)
            y_pred = gbr.predict(x_test)
            acc_gbr.append(accuracy(y_pred,y_test))
            rms_gbr.append(rmse(y_pred,y_test))
        print('knn',np.mean(acc_kr),np.mean(rms_kr))
        print('tree',np.mean(acc_gbr),np.mean(rms_gbr))
        print('svm',np.mean(acc_svm),np.mean(rms_svm))


# In[16]:


#search best parameters
params = {'loss' :['ls','lad','huber'],'max_depth':[3,4,5],'max_features':['auto','sqrt','log2']}
gbr = GradientBoostingRegressor()
gbr_cv = GridSearchCV(gbr,params,scoring= metrics.make_scorer(rmse),cv = 5 )
gbr_cv.fit(X,y)


# In[17]:


#drop useless feature density
methods(range(11),X,y)
methods([0,1,2,3,4,5,6,8,7,10],X,y)


# gradientboosting method is best,and accuracy is 0.8763,rmse is 0.7658

# In[18]:


mean_test_score = gbr_cv.cv_results_['mean_test_score']
print(np.min(mean_test_score))
index = np.argmin(mean_test_score)
print(gbr_cv.cv_results_['params'][index])


# In[19]:


#search best parameters for svm
params = {'C' :[3,8,10],'gamma':[0.01,0.1,1],'epsilon':[0.01,0.1,1]}
svm = SVR()
svm_cv = GridSearchCV(svm,params,scoring= metrics.make_scorer(rmse),cv = 5 )
svm_cv.fit(X,y)


# In[20]:


mean_test_score = svm_cv.cv_results_['mean_test_score']
print(np.min(mean_test_score))
index = np.argmin(mean_test_score)
print(svm_cv.cv_results_['params'][index])


# In[21]:


#search best parameters for knn
params = {'n_neighbors' :[20,15,10]}
kr = KNeighborsRegressor()
kr_cv = GridSearchCV(kr,params,scoring= metrics.make_scorer(rmse),cv = 5 )
kr_cv.fit(X,y)


# In[22]:


mean_test_score = kr_cv.cv_results_['mean_test_score']
print(np.min(mean_test_score))
index = np.argmin(mean_test_score)
print(kr_cv.cv_results_['params'][index])


# In[23]:


#feature corelations
white_data_heatmap = white_data.iloc[:,:-1]
white_data_heatmap = white_data_heatmap.corr()
sns.heatmap(white_data_heatmap)
plt.show()


# In[24]:


sns.pairplot(white_data,vars = white_data.columns[:-1])
plt.show()


# # Red wine analysis

# In[25]:


red_data = pd.read_csv('winequality-red.csv',sep = ';')
red_data.info()


# In[26]:


sns.distplot(red_data.quality)
plt.show()


# In[27]:


#standardlize data
for col in red_data.columns[:-1]:
    red_data[col] = (red_data[col]-red_data[col].min())/(red_data[col].max()-red_data[col].min())


# In[28]:


#remove outliers
for col in red_data.columns[:-1]:
    red_data = red_data[red_data[col]<0.9]
len(red_data)


# In[29]:


X_red = red_data.iloc[:,:-1].values
y_red = red_data.iloc[:,-1].values


# In[30]:


baseline(X_red,y_red)


# In[31]:


#search best parameters
params = {'loss' :['ls','lad','huber'],'max_depth':[3,4,5],'max_features':['auto','sqrt','log2']}
gbr_red = GradientBoostingRegressor()
gbr_cv_red = GridSearchCV(gbr_red,params,scoring= metrics.make_scorer(rmse),cv = 5 )
gbr_cv_red.fit(X_red,y_red)


# In[32]:


#drop useless feature density
methods(range(11),X_red,y_red)
#methods([1,2,3,4,6,8,7,10],X_red,y_red)


# In[33]:


mean_test_score = gbr_cv_red.cv_results_['mean_test_score']
print(np.min(mean_test_score))
index = np.argmin(mean_test_score)
print(gbr_cv_red.cv_results_['params'][index])


# In[34]:


#search best parameters for svm
params = {'C' :[3,8,10],'gamma':[0.01,0.1,1],'epsilon':[0.01,0.1,1]}
svm_red = SVR()
svm_cv_red = GridSearchCV(svm_red,params,scoring= metrics.make_scorer(rmse),cv = 5 )
svm_cv_red.fit(X_red,y_red)


# In[35]:


mean_test_score = svm_cv_red.cv_results_['mean_test_score']
print(np.min(mean_test_score))
index = np.argmin(mean_test_score)
print(svm_cv_red.cv_results_['params'][index])


# In[36]:


#search best parameters for knn
params = {'n_neighbors' :[20,15,10]}
kr_red = KNeighborsRegressor()
kr_cv_red = GridSearchCV(kr_red,params,scoring= metrics.make_scorer(rmse),cv = 5 )
kr_cv_red.fit(X_red,y_red)


# In[37]:


mean_test_score = kr_cv_red.cv_results_['mean_test_score']
print(np.min(mean_test_score))
index = np.argmin(mean_test_score)
print(kr_cv_red.cv_results_['params'][index])


# In[38]:


feature = range(11)
gbr = GradientBoostingRegressor(max_depth = 3,max_features = 'sqrt',loss = 'huber')
svm = SVR(C= 3, epsilon= 0.1, gamma= 1)
kr = KNeighborsRegressor(n_neighbors = 15)
kf = KFold(n_splits=5,shuffle = True)
for i in range(1):
    x = X_red[:,feature]
    acc_gbr = []
    rms_gbr = []
    acc_kr = []
    rms_kr = []
    for train_index,test_index in kf.split(x):
        x_train = x[train_index]
        y_train = y_red[train_index]
        x_test = x[test_index]
        y_test = y_red[test_index]
        gbr.fit(x_train,y_train)
        y_pred = gbr.predict(x_test)
        acc_gbr.append(accuracy(y_pred,y_test))
        rms_gbr.append(rmse(y_pred,y_test))
        kr.fit(x_train,y_train)
        y_pred = kr.predict(x_test)
        acc_kr.append(accuracy(y_pred,y_test))
        rms_kr.append(rmse(y_pred,y_test))
    print('tree',np.mean(acc_gbr),np.mean(rms_gbr))
    print('knn',np.mean(acc_kr),np.mean(rms_kr))


# In[39]:


red_data_heatmap = red_data.iloc[:,:-1]
red_data_heatmap = red_data_heatmap.corr()
sns.heatmap(red_data_heatmap)
plt.show()


# In[40]:


sns.pairplot(red_data,vars = red_data.columns[:-1])
plt.show()


# # white and red wine analysis

# In[41]:


red_data.columns


# In[42]:


white_data.columns


# In[43]:


white_data['color'] = 1


# In[44]:


red_data['color'] = 0


# In[45]:


all_data = white_data.append(red_data)
len(all_data)


# In[46]:


sns.distplot(all_data.quality)
plt.show()


# In[47]:


x_index = [0,1,2,3,4,5,6,7,8,9,10,12]
X_all = all_data.values[:,x_index]
y_all = all_data.values[:,-2]


# In[48]:


baseline(X_all,y_all)


# In[49]:


#search best parameters
params = {'loss' :['ls','lad','huber'],'max_depth':[3,4,5],'max_features':['auto','sqrt','log2']}
gbr_all= GradientBoostingRegressor()
gbr_cv_all = GridSearchCV(gbr_all,params,scoring= metrics.make_scorer(rmse),cv = 5 )
gbr_cv_all.fit(X_all,y_all)


# In[50]:


methods(range(12),X_all,y_all)


# In[51]:


mean_test_score = gbr_cv_all.cv_results_['mean_test_score']
print(np.min(mean_test_score))
index = np.argmin(mean_test_score)
print(gbr_cv_all.cv_results_['params'][index])


# In[52]:


#search best parameters for svm
params = {'C' :[3,8,10],'gamma':[0.01,0.1,1],'epsilon':[0.01,0.1,1]}
svm_all = SVR()
svm_cv_all = GridSearchCV(svm_all,params,scoring= metrics.make_scorer(rmse),cv = 5 )
svm_cv_all.fit(X_all,y_all)


# In[53]:


mean_test_score = svm_cv_all.cv_results_['mean_test_score']
print(np.min(mean_test_score))
index = np.argmin(mean_test_score)
print(svm_cv_all.cv_results_['params'][index])


# In[54]:


#search best parameters for knn
params = {'n_neighbors' :[20,15,10]}
kr_all = KNeighborsRegressor()
kr_cv_all = GridSearchCV(kr_all,params,scoring= metrics.make_scorer(rmse),cv = 5 )
kr_cv_all.fit(X_all,y_all)


# In[55]:


mean_test_score = kr_cv_all.cv_results_['mean_test_score']
print(np.min(mean_test_score))
index = np.argmin(mean_test_score)
print(kr_cv_all.cv_results_['params'][index])


# In[56]:


feature = range(12)
gbr = GradientBoostingRegressor(max_depth = 5,max_features = 'log2',loss = 'ls')
svm = SVR(C= 8)
kf = KFold(n_splits=5,shuffle = True)
for i in range(1):
    x = X_all[:,feature]
    acc_gbr = []
    rms_gbr = []
    for train_index,test_index in kf.split(x):
        x_train = x[train_index]
        y_train = y_all[train_index]
        x_test = x[test_index]
        y_test = y_all[test_index]
        gbr.fit(x_train,y_train)
        y_pred = gbr.predict(x_test)
        acc_gbr.append(accuracy(y_pred,y_test))
        rms_gbr.append(rmse(y_pred,y_test))
    print('tree',np.mean(acc_gbr),np.mean(rms_gbr))


# In[57]:



all_data_heatmap = all_data.iloc[:,x_index]
all_data_heatmap = all_data_heatmap.corr()
sns.heatmap(all_data_heatmap)
plt.show()


# In[58]:


sns.pairplot(all_data,
vars = all_data.columns[:-2],
hue = 'color')
plt.legend = ['white','red']
plt.show()

