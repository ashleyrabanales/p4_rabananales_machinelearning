#install.packages("tidymodels")
#install.packages("DALEX")
#install.packages("discrim")
#install.packages("naivebayes")
#install.packages("vip")
#install.packages("xgboost")
#install.packages("patchwork") #put ggplot together
#install.packages("GGally")


library(tidyverse)
library(tidymodels)
library(DALEX)
library(skimr)
library(GGally)
library(xgboost)
library(vip)
library(patchwork)

httpgd::hgd()
httpgd::hgd_browse()

dat_ml <- read_rds("dat_ml.rds")

#Training and Testing Data
#The scope of rsample is to provide the basic building blocks for creating and analyzing resamples of a data set, but this package does not include code for modeling or calculating statistics.

#rsample
#initial_split()
#training() = testing()

set.seed(76) #setting a seed for the start of the session, random happens is sequence from the seed
dat_split <- initial_split(dat_ml, prop = 2/3, strata = before1980)
#strat making sure the balance of an opject in testing and training, to pull it out.
#run it with the seed

dat_train <- training(dat_split)
dat_test <- testing(dat_split)

#Tidymodels and our model development
#We need a few more R packages as we use tidymodels.



#Model fit - The goal of parsnip is to provide a tidy, unified interface to models that can be used to try a range of models without getting
#bogged down in the syntactical minutiae of the underlying packages.

#Two mode: reg and class

#3 built model to run prediction
bt_model <- boost_tree() %>%
    set_engine(engine = "xgboost") %>%
    set_mode("classification") %>%
    fit(before1980 ~ ., data = dat_train)

logistic_model <- logistic_reg() %>%
    set_engine(engine = "glm") %>%
    set_mode("classification") %>%
    fit(before1980 ~ ., data = dat_train)

nb_model <- discrim::naive_Bayes() %>%
    set_engine(engine = "naivebayes") %>%
    set_mode("classification") %>%
    fit(before1980 ~ ., data = dat_train)

#Boosted - 

#logistic -

  #  Feature importance - vip is an R package for constructing variable importance plots (VIPs). VIPs are part of a larger 
  #framework referred to as interpretable machine learning (IML)
vip(bt_model, num_features = 20) + vip(logistic_model, num_features = 20)

#Evaluating our predictions- yardstick is a package to estimate how well models are working using tidy data principles.
#
preds_logistic <- bind_cols(
    predict(logistic_model, new_data = dat_test),
    predict(logistic_model, dat_test, type = "prob"),
    truth = pull(dat_test, before1980)
  )

# takes a minute
preds_nb <- bind_cols(
    predict(nb_model, new_data = dat_test),
    predict(nb_model, dat_test, type = "prob"),
    truth = pull(dat_test, before1980)
  )

preds_bt <- bind_cols(
    predict(bt_model, new_data = dat_test),
    predict(bt_model, dat_test, type = "prob"),
    truth = pull(dat_test, before1980)
  )

##Now, we can evaluate our prediction performance.
#conf_mat()
#metrics()
#precision()
#metric_set()
#roc_curve()
#autoplot()
#let’s combine all three pred_ dataframes into one for combined summaries.