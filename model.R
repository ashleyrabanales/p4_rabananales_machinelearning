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

  #  Feature importance - vip is an R package for constructing variable importance plots (VIPs). VIPs are part of a larger 
  #framework referred to as interpretable machine learning (IML)
(vip(bt_model, num_features = 20) + labs(title = "boosted")) + 
(vip(logistic_model, num_features = 20) + labs(title = "logistic"))


#Evaluating our predictions- yardstick is a package to estimate how well models are working using tidy data principles.
#
preds_logistic <- bind_cols(
    predict(logistic_model, new_data = dat_test),
    predict(logistic_model, dat_test, type = "prob"),
    truth = pull(dat_test, before1980)
  )

# takes a minute
preds_logistic <- bind_cols(
    predict(logistic_model, new_data = dat_test),
    predict(logistic_model, dat_test, type = "prob"),
    truth = pull(dat_test, before1980)
  )

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

preds_bt %>% conf_mat(truth, .pred_class)
preds_nb %>% conf_mat(truth, .pred_class)
preds_logistic %>% conf_mat(truth, .pred_class)

metrics_calc <- metric_set(accuracy, bal_accuracy, precision, recall, f_meas)

preds_bt %>%
    metrics_calc(truth, estimate = .pred_class)

preds_nb %>%
    metrics_calc(truth, estimate = .pred_class)

preds_bt %>%
    roc_curve(truth, estimate = .pred_before) %>%
    autoplot()

preds_nb %>%
    roc_curve(truth, estimate = .pred_before) %>%
    autoplot()

preds_bt %>% metrics(truth, .pred_class)
preds_all <- bind_rows(
        mutate(preds_nb, model = "Naive Bayes"),
        mutate(preds_bt, model = "Boosted Tree"),
        mutate(preds_logistic, model = "Logistic Regression")
)

preds_all %>%
    group_by(model) %>%
    roc_curve(truth, estimate = .pred_before) %>%
    autoplot()

preds_all %>%
    group_by(model) %>%
    metrics_calc(truth, estimate = .pred_class) %>%
    pivot_wider(names_from = .metric, values_from = .estimate)

##Now, we can evaluate our prediction performance.
#conf_mat()
#metrics()
#precision()
#metric_set()
#roc_curve()
#autoplot()
#let’s combine all three pred_ dataframes into one for combined summaries.