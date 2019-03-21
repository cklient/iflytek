
# coding: utf-8

# In[ ]:


import datetime
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
import lightgbm as lgb
import warnings
import time
import pandas as pd
import numpy as np
import os 


# In[ ]:


train = pd.read_hdf('../data/train_data2.hdf', key='xunfei')

test = pd.read_hdf('../data/test_data2.hdf', key='xunfei')
data = pd.concat([train, test], axis=0, ignore_index=True)


# In[ ]:


subs=pd.read_csv('../subs/10151.csv')
oof=pd.read_hdf('../oof/10151.csv')


# In[ ]:


oof['predicted_score']=oof['oof'].values
del oof['oof']


# In[ ]:


oof_subs=oof.append(subs).reset_index(drop=True)


# In[ ]:


train_stack=pd.read_csv('../data/stacking_train.csv')
test_stack=pd.read_csv('../data/stacking_pred.csv')


# In[ ]:


stack=train_stack.append(test_stack).reset_index(drop=True)


# In[ ]:


stack=stack.merge(oof_subs,on='instance_id',how='left')


# In[ ]:


stack.to_csv('../data/stacking.csv',index=None)

