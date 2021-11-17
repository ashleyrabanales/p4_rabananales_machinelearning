#%%
# import sys
# !{sys.executable} -m pip install scikit-learn
#%%
# import sys
# !{sys.executable} -m pip install numpy scipy matplotlib ipython scikit-learn pandas pillow
#%%
import pandas as pd
import numpy as np
#%%

from plotnine import *  
#%%

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
# %%

#tranactional data - data of register from purchase of a customer.

 dat = pd.read_csv('SalesBook_2013.csv')

 # %%
 # select variables we will use in class.
 # drop homes that are not single family or duplexes
 dat_ml = (dat
     .filter(['NBHD', 'PARCEL', 'LIVEAREA', 'FINBSMNT',  
         'BASEMENT', 'YRBUILT', 'CONDITION', 'QUALITY',
         'TOTUNITS', 'STORIES', 'GARTYPE', 'NOCARS',
         'NUMBDRM', 'NUMBATHS', 'ARCSTYLE', 'SPRICE',
         'DEDUCT', 'NETPRICE', 'TASP', 'SMONTH',
         'SYEAR', 'QUALIFIED', 'STATUS'])
     .rename(columns=str.lower)
     .query('totunits <= 2'))
# %%
dat_ml # to check the filter data
#Write a short paragraph describing your data. What does a row represent? What measures do we have for each row?
#each row is a sale 
##Each row represents an individual realestate sale, with with columns involving detail of size, quality, price, and etc.
##Predict the house build before 1980? yes or no?
# %%
#Creating our target variable
#np.where()
#assign()
dat_ml = (dat_ml
    .query('yrbuilt != 0 & condition!="None"') #house after 1980 is 0
    .assign( 
    before1980 = lambda x: np.where(x. yrbuilt < 1980,
    1, 0)))
#lambda sets the x in making the vari built before 1980 as 1 and if not then 0
#regression problme to classification, i n bulding y- varaibles bc we are finding the year of the house is before 1980.
#house built before 1980 is 1
# %%
dat_ml.yrbuilt.value_counts()
# %%
#what to be useful? quality or bathrooms?

(ggplot(dat_ml, aes(x='numbaths.astype(str)', y = 'yrbuilt')) +
    geom_boxplot())
#%%
#or

(ggplot(dat_ml, aes(x='quality', y = 'yrbuilt')) +
    geom_boxplot())
#%%



dat_ml.quality.value_counts()
replace_dictionary = {
    "E-": -0.3, "E": 0, "E+":0.3 ,  
    "D-": 0.7,"D":1 ,"D+":1.3, 
    "C-": 1.7, "C": 2, "C+": 2.3,
    "B-": 2.7, "B": 3, "B+": 3.3,
    "A-":3.7, "A-":4, "A+":4.3,
    "X-":4.7, "X":5, "X-": 5.5
}

qual_order =dat_ml.quality.replace(replace_dictionary)
#%%
#one-hot-encode or dummy variables
#arcstyle
#fixing ordinal and nominal?
#What nominal variables do we have? Which have too many categories?
#pd.get_dummies()
#.fillna()
##.str.contains()

#What ordinal variables do we have?
#.replace()
##.astype('float')

#%%
dat_ml.quality.value_counts()
replace_dictionary = {
    "Excel":3,
    "VGood":2,
    "Good":1,
    "AVG":0,
    "Avg":0,
    "Fair":-1,
    "Poor": -2
}
cond_ord = dat_ml.condition.replace(replace_dictionary)
# %%


#one-hot-encode or dummy variables
#arrcstyle, garge type, ndhd
dat_ml.arcstyle.value_counts()
dat_ml.gartype.value_counts()

# %%
#neighbor too many, drop
#drop_first - the column to create a defaults 
pd.get_dummies(dat_ml.filter(['arcstyle']),
drop_first=True)

# %%
