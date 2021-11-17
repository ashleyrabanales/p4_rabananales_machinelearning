#%%
import sys
!{sys.executable} -m pip install scikit-learn
#%%
# import sys
!{sys.executable} -m pip install numpy scipy matplotlib ipython scikit-learn pandas pillow
#%%
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
# %%
