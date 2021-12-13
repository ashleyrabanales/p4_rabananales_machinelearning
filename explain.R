library(tidyverse)
library(tidymodels)
library(DALEX)
library(vip)
library(patchwork)


httpgd::hgd()
httpgd::hgd_browse()

dat_ml <- read_rds("dat_ml.rds")%>%
    select(arcstyle_ONE.AND.HALF.STORY, arcstyle_ONE.STORY, numbaths,
        tasp, livearea, basement, condition, stories, quality, before1980) %>%
    filter(livearea < 5500) # 99th percentile is 5429.04


set.seed(76)
dat_split <- initial_split(dat_ml, prop = 2/3, strata = before1980)

dat_train <- training(dat_split)
dat_test <- testing(dat_split)
#before is now 0 and after is 1
dat_exp <- mutate(dat_train, before1980 = as.integer(dat_train$before1980) - 1)

head(dat_exp$before1980)
head(dat_train$before1980)

#Model fit
#The naive Bayes model didn’t work well. Let’s just use the logistic and xgboost model.
bt_model <- boost_tree() %>%
    set_engine(engine = "xgboost") %>%
    set_mode("classification") %>%
    fit(before1980 ~ ., data = dat_train)

logistic_model <- logistic_reg() %>%
    set_engine(engine = "glm") %>%
    set_mode("classification") %>%
    fit(before1980 ~ ., data = dat_train)

vip(bt_model)
vip(logistic_model)


#DALEX: Model explanation
#Notice that we have to remove our target and then use the 0/1 target from dat_exp.
explainer_bt <- DALEX::explain(
    bt_model,
    select(dat_exp, -before1980), dat_exp$before1980, label = "Boosted Trees")

explainer_logistic <- DALEX::explain(
    logistic_model,
    select(dat_exp, -before1980), dat_exp$before1980, label = "Logistic Regression")
#as far as down as possible, predicts the probability of the homes.
#


#DALEX: Model performance
#We can use the two model residual plots to compare our two models.
performance_logistic <- model_performance(explainer_logistic)
performance_bt <- model_performance(explainer_bt)

plot(performance_bt, performance_logistic)
plot(performance_bt, performance_logistic, geom = "boxplot")
#looking at the residual area from 0-1, to know the variety of it and the low on the variabjce,
#low in the blue boxplot.


#We can also look at the feature importance comparison as well. Notice that the feature 
#importance and ordering are a bit different than those shown by vip(). We have defined the loss
#_function that is different from the default.
logistic_parts <- model_parts(explainer_logistic, 
    loss_function = loss_root_mean_square)
bt_parts <- model_parts(explainer_bt,
    loss_function = loss_root_mean_square)

plot(logistic_parts, bt_parts, max_vars = 10)
#feature importance parts
#logistic model 
#dash  line is a root mean sq error

#blue
#the best fit for it, boosted 

logistic_parts <- model_parts(explainer_logistic, 
    loss_function = loss_root_mean_square, type = "difference")
bt_parts <- model_parts(explainer_bt,
    loss_function = loss_root_mean_square, type = "difference")

plot(logistic_parts, bt_parts, max_vars = 10)
#read the book



#DALEX: Explain score
#library(patchwork)
onehouse_before <- predict_parts(explainer_bt,
    new_observation = select(dat_exp, -before1980) %>%
        dplyr::slice(13800), type = "break_down")

onehouse_after <- predict_parts(explainer_bt,
    new_observation = select(dat_exp, -before1980) %>%
        dplyr::slice(8), type = "break_down")

plot(onehouse_after) + plot(onehouse_before)

dat_train %>% dplyr::slice(c(8, 13800))

#Break Down Profile, on left is after and p=.581 and right is p=.283
#dot line avg of prob
# the probability of being a before and after 1980
#waterfall plots- avg of the dataset and shift.
#rig

#Shap plot
onehouse_before <- predict_parts(explainer_bt,
    new_observation = select(dat_exp, -before1980) %>%
        dplyr::slice(13800), type = "shap")

onehouse_after <- predict_parts(explainer_bt,
    new_observation = select(dat_exp, -before1980) %>%
        dplyr::slice(8), type = "shap")

plot(onehouse_after) + plot(onehouse_before)

profile_bt <- model_profile(explainer_bt)
profile_explainer <- model_profile(explainer_logistic)

plot(profile_bt, profile_explainer)
#size of the green bar has a more impact on the left(after) affecting by the outcome of the variables(part of it..)

