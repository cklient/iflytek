
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


pd.set_option('display.max_columns',100)
data['time']=(data['timee']-data['timee'].min())/(data['timee'].max()-data['timee'].min())


# In[ ]:


ad_cate_feature=['adid','advert_id','orderid','campaign_id','creative_id','creative_tp_dnf','creative_type',
       'creative_is_jump','creative_is_download','creative_has_deeplink','advert_industry_inner0','advert_industry_inner1']
media_cate_feature=['app_cate_id','f_channel','app_id','inner_slot_id']
content_cate_feature=['city', 'carrier', 'province', 'nnt', 'devtype', 'osv', 'make', 'model']

all_data=data.copy()


need=list(all_data.index)


# 循环读取
import gc
from tqdm import tqdm#'unsparse','timesparse',,'expsparse1','expsparse2','dupsparse','innersparse1','dupsparse1'
filelist=['expsparsex','expsparse1','innersparse1','appearsparses']#'svdsparsenew',
start=1
tail=0
##all_df=sparse.csr_matrix((len(all_data),0))
maxrec={}
for f in filelist:
   
    ttf=pd.read_hdf( '../data/'+f+'.hdf').astype(np.float32).iloc[need,:]
    orgshape=ttf.shape[1]
   
    
    '''use=pd.read_hdf('../data/'+f+'_catvip'+'.hdf',key='xunfei')['catvip'].values
    
    ttf=ttf.iloc[:,use]
    maxrec[f]=use#ttf.shape[1]'''
    if (start==1):
        all_df=ttf.copy()
        start=0
    else:
        all_df=pd.concat([all_df,ttf],axis=1)
    print(f,orgshape,(ttf.shape[1])/orgshape)        
    del ttf
    gc.collect()


# In[ ]:


from tqdm import tqdm
num_feature = ['creative_width', 'creative_height', 'hour','user_tags_len']#,'time'
ttmp=data[num_feature].values.astype(np.int32)[need,:]
for col in tqdm(range(ttmp.shape[1])):
    all_df['num'+str(col)]=ttmp[:,col]

del ttmp
gc.collect()


# In[ ]:


# 默认加载 如果 增加了cate类别特征 请改成false重新生成
import os
if os.path.exists('../feature/base_train_csr.npz') and True:
    print('load_csr---------')
    base_train_csr = sparse.load_npz( '../feature/base_train_csr.npz').tocsr().astype('bool')
    base_predict_csr = sparse.load_npz('../feature/base_predict_csr.npz').tocsr().astype('bool')
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

    cv = CountVectorizer(min_df=10)
    for feature in ['user_tagss']:
        data[feature] = data[feature].astype(str)
        cv.fit(data[feature])
        base_train_csr = sparse.hstack((base_train_csr, cv.transform(train_x[feature].astype(str))), 'csr', 'bool')
        base_predict_csr = sparse.hstack((base_predict_csr, cv.transform(predict_x[feature].astype(str))), 'csr',
                                         'bool')
    print('cv prepared !')

    sparse.save_npz( '../feature/base_train_csr.npz', base_train_csr)
    sparse.save_npz('../feature/base_predict_csr.npz', base_predict_csr)


# In[ ]:


#catvip=pd.read_hdf('../data/catvip.hdf', key='xunfei')['catvip'].values
allbase=sparse.vstack((base_train_csr,base_predict_csr),'csr')
#allbase=allbase[need,:][:,catvip]
#allbase=allbase.toarray()


# In[ ]:


all_df=sparse.hstack((allbase,all_df),'csr')


# In[ ]:


train_x =all_df[:-len(test)]
train_y = data[data.target != -1].target.values


# In[ ]:


test_x=all_df[-len(test):]


# In[ ]:



import gc
del allbase
gc.collect()
del all_df
gc.collect()
del data
gc.collect()


# In[ ]:


del all_data
gc.collect()


# In[ ]:



import datetime
import lightgbm as lgb
import gc
from scipy.sparse import csr_matrix
from sklearn.model_selection import KFold, StratifiedKFold
# sklearn风格lightgbm
import random
fea_rec=[]
fea_vip=[]
sub_preds = pd.DataFrame()

oof_predss = np.zeros(train_x.shape[0])
#argsDict={'bagging_fraction': 0.7441379310344828, 'bagging_freq': 2, 'feature_fraction': 0.8489655172413793, 'learning_rate': 0.07241379310344827, 'max_bin': 268, 'max_depth': -1, 'min_child_weight': 1.3684210526315788, 'min_data_in_bin': 10, 'min_split_gain': 0.21052631578947367, 'num_boost_round': 5000, 'num_leaves': 50, 'rands': 396, 'reg_alpha': 1.8421052631578947, 'reg_lambda': 7.894736842105263, 'scale_pos_weight': 1.0}

argsDict={'bagging_fraction': 0.98, 'bagging_freq': 8, 'feature_fraction': 0.9027586206896552, 'learning_rate': 0.052631578947368425, 'max_bin': 260, 'max_depth': -1, 'min_child_weight': 0.1724137931034483, 'min_data_in_bin': 7, 'min_split_gain': 0.1, 'num_boost_round': 5000, 'num_leaves': 56, 'rands': 152, 'reg_alpha': 6.842105263157895, 'reg_lambda': 13.974358974358974, 'scale_pos_weight': 1.0}

LGB=lgb.LGBMClassifier(
        num_leaves=argsDict["num_leaves"],
        max_depth=argsDict["max_depth"],
        learning_rate=argsDict["learning_rate"],
        n_estimators=argsDict['num_boost_round'],
         min_split_gain=argsDict["min_split_gain"],
        #min_child_samples=argsDict["min_data_in_leaf"],
        min_child_weight=argsDict["min_child_weight"],
        subsample=argsDict["bagging_fraction"],
        subsample_freq=argsDict["bagging_freq"],
        colsample_bytree=argsDict["feature_fraction"],
        reg_alpha=argsDict["reg_alpha"],
        reg_lambda=argsDict["reg_lambda"],
        scale_pos_weight=argsDict["scale_pos_weight"],
        
        
        is_training_metric= True,
        boosting_type='gbdt',
        metric='binary_logloss',
        n_jobs=9,
        #n_threads=10,
        seed=argsDict["rands"],
        drop_seed=argsDict["rands"],
        bagging_seed=argsDict["rands"],
        feature_fraction_seed=argsDict["rands"],
        random_state=argsDict["rands"],
        
        max_bin=argsDict["max_bin"],
        min_data_in_bin=argsDict["min_data_in_bin"],
    )
rands=random.randint(0,100)  
skf = StratifiedKFold(n_splits=20, shuffle=True, random_state=argsDict["rands"])
tmprmselist=[]
treerec=[]
for ii,(train_index, test_index) in enumerate(skf.split(train_x, train_y)):
    
    x_train, x_val = train_x[train_index], train_x[test_index]
    y_train, y_val = train_y[train_index], train_y[test_index]
    print('fold:',ii)
    LGB.fit(x_train,y_train.reshape(-1,),eval_set=(x_val,y_val),eval_metric='logloss',early_stopping_rounds=100,verbose=100)
    tmplist=list(LGB.predict_proba(x_val)[:,1])
    oof_predss[test_index] =tmplist
        #print(LGB.evals_result_)
    tmprmse =np.mean(LGB.evals_result_['valid_0']['binary_logloss'])#np.sqrt(mean_squared_error(y_val, tmplist))
    print('best_itor:',LGB.best_iteration_,'    logloss:',tmprmse)    
    treerec.append(LGB.best_iteration_)
    tmprmselist.append(tmprmse)

    sub_preds['estimators'+str(ii)]=(LGB.predict_proba(test_x,num_iteration=LGB.best_iteration_))[:,1]
    
    importance_dict={}
    for col,val in zip(range(train_x.shape[1]),LGB.feature_importances_):
        importance_dict[col]=val
    ser=pd.Series(importance_dict).sort_values(ascending=False)
    fea_vip+=list(ser[ser>3].index)
    del x_train
    gc.collect()
    del x_val
    gc.collect()


# In[ ]:


sub=test[['instance_id']]
sub['predicted_score']=(sub_preds.mean(axis=1)).values
org_test=pd.read_table('../data/round2_iflyad_test_feature.txt')
sub=org_test[['instance_id']].merge(sub,on='instance_id',how='left')
sub.to_csv('../subs/10151.csv',index=None)


# In[ ]:


oof_df=pd.DataFrame()
oof_df['instance_id']=train['instance_id'].values
oof_df['oof']=oof_predss.reshape(-1,)
oof_df.to_hdf('../oof/10151.csv',key='xunfei')

