
# coding: utf-8

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
from scipy import sparse
from scipy.sparse.linalg import svds
import gc


# In[ ]:


train = pd.read_hdf('../data/train_data2.hdf', key='xunfei')
test = pd.read_hdf('../data/test_data2.hdf', key='xunfei')
data = pd.concat([train, test], axis=0, ignore_index=True)


# In[ ]:


allsize=pd.read_hdf('../data/size.hdf', key='xunfei')
data=data.merge(allsize,on='instance_id',how='left')
data['area']=data['creative_width_org']*data['creative_height_org']


# In[ ]:


pd.set_option('display.max_columns',100)
data['time']=(data['timee']-data['timee'].min())/(data['timee'].max()-data['timee'].min())
#data


# In[ ]:


ad_cate_feature=['adid','advert_id','orderid','campaign_id','creative_id','creative_tp_dnf','creative_type','creative_is_jump',
                 'creative_is_download','creative_has_deeplink','advert_industry_inner0','advert_industry_inner1','creative_width', 
                 'creative_height']
media_cate_feature=['app_cate_id','f_channel','app_id','inner_slot_id']
content_cate_feature=['city', 'carrier', 'province', 'nnt', 'devtype', 'osv','os_name' ,'make', 'model']


# In[ ]:


origin_cate_list = ad_cate_feature + media_cate_feature + content_cate_feature


# In[ ]:


cate_feature = origin_cate_list

num_feature = ['creative_width', 'creative_height', 'hour','user_tags_len','time','creative_width_org','creative_height_org','area']


# In[ ]:


feature = cate_feature + num_feature


# In[ ]:


all_data=data.copy()


# In[ ]:


orders = {}

for col in (origin_cate_list+num_feature):
    orders[col] = 10 ** (int(np.log(all_data[col].max() + 1) / np.log(10)) + 1)
def get_group(df, cols):
    
    group = df[cols[0]].copy()
    for col in cols[1:]:
        group = group * orders[col] + df[col]
        
    return group


# In[ ]:


def col_name(cols, func):
    return '_'.join(cols) + '_' + func.__name__


# In[ ]:


def last_time_diff(cols,X):##距离最后一次出现时间间隔
    
    df=all_data.copy()
    group = get_group(df, cols)
        
    last_time = df.groupby(group).timee.last()
    
    return (group.map(last_time) - df.timee).astype(np.float32).values
def last_appear(cols,X):
    df=all_data.copy()
    group = get_group(df, cols)
    last_heard ={}
    result = []
    for t, g in zip(df.timee, group):
        if g in last_heard:
            result.append(t - last_heard[g])
        else:
            result.append(-1)
        last_heard[g] = t
    return result
def next_appear(cols,X):
    df=all_data.copy()
    result = []
    df_reverse = df.sort_index(ascending=False)
    group = get_group(df_reverse,  cols)
    
    next_heard = {}
    for g, t in zip(group, df_reverse.timee):
        if g in next_heard:
            result.append(t - next_heard[g])
        else:
            result.append(-1)
        next_heard[g] = t
    
    result.reverse()
    return result
def extract_last_next_appear():
    X=pd.DataFrame()
    funcs=[last_time_diff]#last_appear,next_appear,
    catcol=origin_cate_list.copy()
    
    ad_cate=[]
    for cc in ad_cate_feature:
        if len(all_data[cc].value_counts())>5:
            ad_cate.append(cc)
    
    ad_cate.remove('adid')
    
    
    catcol.remove('creative_is_jump')
    catcol.remove('creative_is_download')
    catcol.remove('creative_has_deeplink')
    
    for col in catcol:
        col=[col]
        print('last_next_appear:',col)
        for func in funcs:
            X[col_name(col, func)]=func(col,X)
    
    print('extract_last_next_appear done')
    X=X.astype(np.float32)
    X.to_hdf('../data/appearsparses.hdf',key='xunfei')


# In[ ]:


extract_last_next_appear()


# In[ ]:


num_feature = ['hour','user_tags_len','time','creative_width_org','creative_height_org','area']


# In[ ]:


def experiments1_mean_std(col,X):
    tt=all_data.groupby(col)[num_feature].agg('mean').reset_index()
    ttmp=all_data[[col]].merge(tt,on=col,how='left').fillna(-1)
    ttmp=ttmp[num_feature]
    ttmp.columns=[col+'num_mean'+str(i) for i in range(len(num_feature))]
    X=pd.concat([X,ttmp],axis=1)
    
    tt=all_data.groupby(col)[num_feature].agg('std').reset_index()
    ttmp=all_data[[col]].merge(tt,on=col,how='left').fillna(-1)
    ttmp=ttmp[num_feature]
    ttmp.columns=[col+'num_std'+str(i) for i in range(len(num_feature))]
    X=pd.concat([X,ttmp],axis=1)
    
    tt=all_data.groupby(col)[num_feature].agg('min').reset_index()
    ttmp=all_data[[col]].merge(tt,on=col,how='left').fillna(-1)
    ttmp=ttmp[num_feature]
    ttmp.columns=[col+'num_min'+str(i) for i in range(len(num_feature))]
    X=pd.concat([X,ttmp],axis=1)
    
    tt=all_data.groupby(col)[num_feature].agg('max').reset_index()
    ttmp=all_data[[col]].merge(tt,on=col,how='left').fillna(-1)
    ttmp=ttmp[num_feature]
    ttmp.columns=[col+'num_max'+str(i) for i in range(len(num_feature))]
    X=pd.concat([X,ttmp],axis=1)
    return X
def extract_experiments1():
    org=[]
    for col in origin_cate_list:
        if len(all_data[col].value_counts())>5:
            org.append(col)
        
    ad_cate=ad_cate_feature.copy()
    ad_cate.remove('creative_width')
    ad_cate.remove('creative_height')
    ad_cate.remove('creative_is_jump')
    ad_cate.remove('creative_is_download')
    ad_cate.remove('creative_has_deeplink')
   
    #org=origin_cate_list.copy()
    org.remove('creative_width')
    org.remove('creative_height')
    
    X=pd.DataFrame(np.zeros(len(all_data)),columns=['s'])
    for col in org:
        print('extract_experiments1:'+col)
        X=experiments1_mean_std(col,X)
    print('extract_experiments1 done')    
    del X['s']
    #return X
    X.to_hdf('../data/expsparsex.hdf',key='xunfei')


# In[ ]:


extract_experiments1()


# In[ ]:


def experiments2_ount(grp,tar,X):
    cols=[grp]+[tar]
    mapp=all_data[cols].groupby(grp).apply(lambda x: len(set(x[tar].values)))
    tt=all_data[grp].map(mapp).fillna(-1)
    X[col_name(cols, experiments2_ount)+'set']=tt.values
    
    '''mapp=all_data[cols].groupby(grp).apply(lambda x: len(x[tar].values))
    tt=all_data[grp].map(mapp).fillna(-1)
    X[col_name(cols, experiments2_ount)]=tt.values'''
    return X
    
def extract_experiments2():
    X=pd.DataFrame(np.zeros(len(all_data)),columns=['s'])
    ad_cate=ad_cate_feature.copy()
    
    ad_cate.remove('creative_is_jump')
    ad_cate.remove('creative_is_download')
    ad_cate.remove('creative_has_deeplink')
    
    for grp in content_cate_feature+media_cate_feature:
        for tar in ad_cate:
            X=experiments2_ount(grp,tar,X)
    
    for grp in ad_cate:
        for tar in content_cate_feature+media_cate_feature:
            X=experiments2_ount(grp,tar,X)
    
    print('extract_experiments2 done')  
    del X['s']
    X.to_hdf('../data/expsparse1.hdf',key='xunfei')


# In[ ]:


extract_experiments2()


# In[ ]:


def inner_count(grp,tar,X):
    cols=[grp]+[tar]
    mapp=all_data[cols].groupby(grp).apply(lambda x: len(set(x[tar].values)))
    tt=all_data[grp].map(mapp).fillna(-1)
    X[col_name(cols, experiments2_ount)+'set']=tt.values
  
    return X
def extract_inner():
    X=pd.DataFrame(np.zeros(len(all_data)),columns=['s'])
    ad_cate=ad_cate_feature.copy()
    
    ad_cate.remove('creative_is_jump')
    ad_cate.remove('creative_is_download')
    ad_cate.remove('creative_has_deeplink')
    
    ad_cate.remove('adid')
    
    for grp in ad_cate:
        for tar in ['adid']:
            X=inner_count(grp,tar,X)
    for grp in ['province']:
        for tar in ['city']:
            X=inner_count(grp,tar,X)
    
    for grp in ['province']:
        for tar in ['city']:
            X=inner_count(grp,tar,X)
    for grp in ['make', 'model']:
        for tar in ['osv']:
            X=inner_count(grp,tar,X)
    
    print('extract_inner done')  
    del X['s']
    X.to_hdf('../data/innersparse1.hdf',key='xunfei')


# In[ ]:


extract_inner()


# In[ ]:


predict = data[data.target == -1]

train_x = data[data.target != -1]
train_y = data[data.target != -1].target.values

predict_x = predict.drop('target', axis=1)


# In[ ]:



if os.path.exists('../feature/base_train_csr.npz') and True:
    print('load_csr---------')
    
else:
    base_train_csr = sparse.csr_matrix((len(train), 0))
    base_predict_csr = sparse.csr_matrix((len(predict_x), 0))

    enc = OneHotEncoder()
    for feature in cate_feature:
        enc.fit(data[feature].values.reshape(-1, 1))
        base_train_csr = sparse.hstack((base_train_csr, enc.transform(train_x[feature].values.reshape(-1, 1))), 'csr',
                                       'bool')
        base_predict_csr = sparse.hstack((base_predict_csr, enc.transform(predict[feature].values.reshape(-1, 1))),
                                         'csr',
                                         'bool')
    print('one-hot prepared !')

    cv = CountVectorizer(min_df=20)
    for feature in ['user_tagss']:
        data[feature] = data[feature].astype(str)
        cv.fit(data[feature])
        base_train_csr = sparse.hstack((base_train_csr, cv.transform(train_x[feature].astype(str))), 'csr', 'bool')
        base_predict_csr = sparse.hstack((base_predict_csr, cv.transform(predict_x[feature].astype(str))), 'csr',
                                         'bool')
    print('cv prepared !')

    sparse.save_npz( '../feature/base_train_csr.npz', base_train_csr)
    sparse.save_npz('../feature/base_predict_csr.npz', base_predict_csr)

