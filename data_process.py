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

#######################
######################
####################
# %%
import re
import pandas as pd
import numpy as np

from plotnine import *
# %%
# select variables we will use in class.
dat = (pd.read_csv('/Users/ashleyrabanales/Projects_ST/p4_rabananales_machinelearning/SalesBook_2013.csv', low_memory=False)
    .filter(['NBHD', 'PARCEL', 'LIVEAREA', 'FINBSMNT',  
        'BASEMENT', 'YRBUILT', 'CONDITION', 'QUALITY',
        'TOTUNITS', 'STORIES', 'GARTYPE', 'NOCARS',
        'NUMBDRM', 'NUMBATHS', 'ARCSTYLE', 'SPRICE',
        'DEDUCT', 'NETPRICE', 'TASP', 'SMONTH',
        'SYEAR', 'QUALIFIED', 'STATUS'])
    .rename(columns=str.lower)
    # drop homes that are not single family or duplexes
    .query('totunits <= 2 & yrbuilt != 0 & condition != "None"')
    .assign(
        before1980 = lambda x: np.where(x.yrbuilt < 1980, 1, 0),
        gartype = lambda x: np.where(x.gartype.isnull(), "Missing", x.gartype),
        )
    .sort_values(['parcel','syear', 'smonth'], ascending = False)
    .groupby(['parcel'])
    .first() # removes older selling moments of duplicate homes
    .reset_index()
    .drop(['nbhd', # don't want to deal with nbhd 
        'parcel', # don't want to have unique identifier in model
        'status'], # almost all 'I'.
        axis=1)) # don't want to deal with nbhd and dropcing parcel

# %%
arc_dummies = pd.get_dummies(dat.filter(['arcstyle']),
    drop_first=True)

# %%

replace_quality = {
    "E-":-0.3 , "E":0, "E+":0.3,
    "D-":0.7, "D":1, "D+":1.3,
    "C-":1.7, "C":2, "C+":2.3,
    "B-":2.7, "B":3, "B+":3.3,
    "A-":3.7, "A":4, "A+":4.3,
    "X-":4.7, "X":5, "X+":5.3 
}

replace_condition = {
    "Excel":3,
    "VGood":2,
    "Good":1,
    "AVG":0,
    "Avg":0,
    "Fair":-1,
    "Poor":-2
}

values_missing = {
    "basement":0, 
    "nocars": dat.nocars.median(),
    "numbdrm": dat.numbdrm.median(),
    'numbaths': dat.numbaths.median()}

#%%
# dat_ml.qualified.value_counts()
# dat_ml.gartype.str.contains("att", flags=re.IGNORECASE, regex=True).astype(int)

dat_ml = (dat.assign(
    quality = lambda x: x.quality.replace(replace_quality),
    condition = lambda x: x.condition.replace(replace_condition),
    attachedGarage = lambda x: x.gartype.str.contains("att",
        flags=re.IGNORECASE, regex=True).astype(int),
    detachedGarage = lambda x: x.gartype.str.contains("det",
        flags=re.IGNORECASE, regex=True).astype(int),
    carportGaragae = lambda x: x.gartype.str.contains("cp",
        flags=re.IGNORECASE, regex=True).astype(int),
    noGarage = lambda x: x.gartype.str.contains("none",
        flags=re.IGNORECASE, regex=True).astype(int),
    qualified = lambda x: np.where(x.qualified == "Q", 1, 0))
.drop(columns = ['gartype', 'qualified', 'arcstyle'])
.fillna(values_missing))

dat_ml = pd.concat([dat_ml, arc_dummies], axis=1)

# %%
# now fix missing
dat_ml.isnull().sum()/len(dat_ml)*100


# %%
dat_ml.to_pickle('dat_ml.pkl')
# %%
# dat_ml.gartype.value_counts()
#%%
import sys
!{sys.executable} -m pip install scikit-learn dalex shap
# %%
import pandas as pd
import numpy as np
import joblib # to savel ml models
#from plotnine import *

from sklearn import metrics

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
# %%
#Build training and testing data
dat_ml = pd.read_pickle('dat_ml.pkl')
#%%
# Split our data into training and testing sets.
X_pred = dat_ml.drop(['yrbuilt', 'before1980'], axis = 1)
y_pred = dat_ml.before1980
X_train, X_test, y_train, y_test = train_test_split( #function can create 4 with it
    X_pred, y_pred, test_size = .34, random_state = 76)  
#34% data goes in 34 rows in a xtest, 34 value in a y value, another in a 64.

#%%
#Building our models
clfNB = GaussianNB() # pick a model
clfGB = GradientBoostingClassifier() #function method in an object
clfNB.fit(X_train, y_train) #fit a model
clfGB.fit(X_train, y_train)
ypred_clfNB = clfNB.predict(X_test) #predict a model
ypred_clfGB = clfGB.predict(X_test)

#%%
#settting a threshold from .5
ypred_clfGB_prop = clfGB.predict

#Diagnosing our model with sklearn
# %%
metrics.plot_roc_curve(clfGB, X_test, y_test) #receiving operating curve, 
#i sort all the items with the proablility = 1 diagnose the model of the area under the curve

metrics.plot_roc_curve(clfNB, X_test, y_test)
#alright but best for the other graph.


# %%
metrics.confusion_matrix(y_test, ypred_clfNB)
#%%
metrics.confusion_matrix(y_test, ypred_clfGB)


#Now we can build our own feature importance chart.
# %%
df_features = pd.DataFrame(
    {'f_names': X_train.columns, 
    'f_values': clfGB.feature_importances_}).sort_values('f_values', ascending = False).head(12)

# Python sequence slice addresses 
# can be written as a[start:end:step]
# and any of start, stop or end can be dropped.
# a[::-1] is reverse sequence.
f_names_cat = pd.Categorical(
    df_features.f_names,
    categories=df_features.f_names[::-1])

df_features = df_features.assign(f_cat = f_names_cat)

#import plotnine as *
(ggplot(df_features,
    aes(x = 'f_cat', y = 'f_values')) +
    geom_col() +
    coord_flip() +
    theme_bw()
    )
#Finalizing our model
#We have ignored model tuning beyond variable reduction. You would want to tune your model parameters and evaluate your classification threshold as well.
#Variable reduction
# build reduced model
#%%
compVars = df_features.f_names[::-1].tolist() #flipping the order - REORDER THE STRUCTURE

X_pred_reduced = dat_ml.filter(compVars, axis = 1)
y_pred = dat_ml.before1980

X_train_reduced, X_test_reduced, y_train, y_test = train_test_split(
    X_pred_reduced, y_pred, test_size = .34, random_state = 76)  

clfGB_reduced = GradientBoostingClassifier()
clfGB_reduced.fit(X_train_reduced, y_train)
ypred_clfGB_red = clfGB_reduced.predict(X_test_reduced)

# %%
print(metrics.classification_report(ypred_clfGB_red, y_test)) #standard value used
metrics.confusion_matrix(y_test, ypred_clfGB_red)

#finding the default, using the rough avg.
# %%
#parameter tuning - fitting with using different parameter to see which will fit better
#saving the models
joblib.dump(clfNB, 'models/clfNB.pkl')
joblib.dump(clfGB, 'models/clfGB.pkl')
joblib.dump(clfGB_reduced, 'models/clfGB_final.pkl')
df_features.f_names[::-1].to_pickle('models/compVars.pkl')
#creating a models folder to save it
# %%

