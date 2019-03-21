
# coding: utf-8

import pandas as pd
import numpy as np
pd.set_option('display.max_columns',100)
train1=pd.read_table('../data/round1_iflyad_train.txt')
test1=pd.read_table('../data/round1_iflyad_test_feature.txt')

train2=pd.read_table('../data/round2_iflyad_train.txt')
test2=pd.read_table('../data/round2_iflyad_test_feature.txt')


train=train1.append(train2).drop_duplicates().reset_index(drop=True)
test=test1.append(test2).drop_duplicates().reset_index(drop=True)
del train['os']
del test['os']

train=train.sort_values(by='time').reset_index(drop=True)
test=test.sort_values(by='time').reset_index(drop=True)


test['click']=-1


concat=train.append(test).reset_index(drop=True)
concat=concat.sort_values(by='time').reset_index(drop=True)



concat['advert_industry_inner0']=concat['advert_industry_inner'].apply(lambda x:x.split('_')[0])
concat['advert_industry_inner1']=concat['advert_industry_inner'].apply(lambda x:x.split('_')[1])
del concat['advert_industry_inner']



import time, datetime
concat['mon']=concat['time'].apply(lambda x:(time.localtime(x)).tm_mon)
concat['day']=concat['time'].apply(lambda x:(time.localtime(x)).tm_mday)
concat['hour']=concat['time'].apply(lambda x:(time.localtime(x)).tm_hour)
concat['wday']=concat['time'].apply(lambda x:(time.localtime(x)).tm_wday)
concat['timee']=concat['time']
del concat['time']




rsav=concat[['instance_id','creative_width','creative_height']]
rsav.columns=['instance_id','creative_width_org','creative_height_org']
rsav.to_hdf('../data/size.hdf',key='xunfei')



concat['user_tags']=concat['user_tags'].astype(str)



concat['user_tags_len']=concat['user_tags'].apply(lambda x:len(x.split(',')))
alltags=concat['user_tags'].apply(lambda x:((x.split(','))))
alltagss=alltags.apply(lambda x:' '.join(x))



del concat['app_paid']
del concat['creative_is_voicead']
del concat['creative_is_js']



cool=list(concat.select_dtypes(include='bool').columns)
concat[cool]=(concat[cool]*1).values



#处理手机厂家，型号
concat['make']=concat['make'].apply(lambda x:str(x).lower())
concat['model']=concat['model'].apply(lambda x:str(x).lower())


def complete_col(xh,source='model',tarcol='make'):
    #print(xh) 
    x=concat.loc[xh]
    if (x[tarcol]=='nan'):
        if x[source]!='nan':
            if len(x[source].split())>0:
                return x[source].split()[0]
            else:
                return 'nan'
        else:
            return 'nan'
    else :
        return x[tarcol]


makecom=pd.Series(range(len(concat))).apply(lambda x:complete_col(x,source='model',tarcol='make'))
concat['make']=makecom
concat['make']=concat['make'].apply(lambda x:x.replace('%2522',' ').strip())


def get_make(x):
    #print(x)
    if '%' in x:
        try :return x.split('%')[0].split()[0]
        except:return 'nan'
    elif ',' in x:
        try :return x.split(',')[0].split()[0]
        except:return 'nan'
    elif '+' in x:
        try:return x.split('+')[0].split()[0]
        except:return 'nan'
    elif '-' in x:
        try:return x.split('-')[0].split()[0]
        except:return 'nan'
    elif '.' in x:
        try:return x.split('.')[0].split()[0]
        except:return 'nan'
    else :
        try:return x.split()[0]
        except:return 'nan'


concat['make']=concat['make'].apply(lambda x:get_make(x))
concat['make']=concat['make'].replace('mi','xiaomi').replace('sm','samsung')


def get_mke(x):
    for mak in make_dict:
        if mak in x:
            return mak
    return x


make_dict=list(concat['make'].value_counts()[:100].index)
concat['make']=concat['make'].apply(lambda x:get_mke(x))


tmp=concat['make'].value_counts()==1
other=list(tmp[tmp>0].index)
concat['make']=concat['make'].apply(lambda x:x if x not in other else 'other')


concat['model']=concat['model'].apply(lambda x:x.strip().replace('+',' ').replace(',','').replace('-',' ').replace('.',''))


tmp=concat['model'].value_counts()==1
other=list(tmp[tmp>0].index)
concat['model']=concat['model'].apply(lambda x:x if x not in other else 'other')


# 处理osv
import re
concat['osv']=concat['osv'].fillna('nan9998').astype(str)

concat['osv']=concat['osv'].apply(lambda x:re.sub(r'\D', "", (x)))
tmp=concat['osv'].value_counts()==1
other=list(tmp[tmp>0].index)
concat['osv']=concat['osv'].apply(lambda x:x if x not in other else 'other')


catcol=list(concat.loc[:,:'advert_name'])+['advert_industry_inner0','advert_industry_inner1']
catcol.remove('user_tags')
catcol.remove('instance_id')


from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()

for col in catcol:
    concat[col]=concat[col].astype(str)
    concat[col]=enc.fit_transform(concat[col].fillna(-1))

cool=list(concat.select_dtypes(include='int64').columns)
cool.remove('instance_id')
cool.remove('timee')


concat[cool]=concat[cool].astype(np.int32)


del concat['advert_name']


concat['user_tagss']=concat['user_tags']
del concat['user_tags']


#处理user_tags
concat['user_tagss']=concat['user_tagss'].apply(lambda x:x.split(',')).apply(lambda x:' '.join(x)).apply(lambda x:x.split()).apply(lambda x:' '.join(x))


concat['target']=concat['click']
del concat['click']


train_data = concat[concat['target']!=-1]
test_data = concat[concat['target']==-1]


train_data.to_hdf('../data/train_data2.hdf', key='xunfei')
test_data.to_hdf('../data/test_data2.hdf', key='xunfei')


concat['instance_id'].to_hdf('../data/allid2.hdf', key='xunfei')

