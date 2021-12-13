
library(tidyverse)
library(tidymodels)
library(visdat)
library(skimr)
library(yardstick)

dat <- read_csv("SalesBook_2013.csv") %>%
  select(NBHD, PARCEL, LIVEAREA, FINBSMNT,
         BASEMENT, YRBUILT, CONDITION, QUALITY,
         TOTUNITS, STORIES, GARTYPE, NOCARS,
         NUMBDRM, NUMBATHS, ARCSTYLE, SPRICE,
         DEDUCT, NETPRICE, TASP, SMONTH,
         SYEAR, QUALIFIED, STATUS) %>%
  rename_all(str_to_lower) %>%
  filter(
    totunits <= 2,
    yrbuilt != 0,
    condition != "None") %>%
  mutate(
    before1980 = ifelse(yrbuilt < 1980, "before", "after") %>%factor(levels = c("before", "after")),
    quality = case_when(
      quality == "E-" ~ -0.3, quality == "E" ~ 0,
      quality == "E+" ~ 0.3, quality == "D-" ~ 0.7, 
      quality == "D" ~ 1, quality == "D+" ~ 1.3,
      quality == "C-" ~ 1.7, quality == "C" ~ 2,
      quality == "C+" ~ 2.3, quality == "B-" ~ 2.7,
      quality == "B" ~ 3, quality == "B+" ~ 3.3,
      quality == "A-" ~ 3.7, quality == "A" ~ 4,
      quality == "A+" ~ 4.3, quality == "X-" ~ 4.7,
      quality == "X" ~ 5, quality == "X+" ~ 5.3) , 
    
    condition = case_when(
      condition == "Excel" ~ 3,
      condition == "VGood" ~ 2,
      condition == "Good" ~ 1,
      condition == "AVG" ~ 0,
      condition == "Avg" ~ 0,
      condition == "Fair" ~ -1,
      condition == "Poor" ~ -2),
    arcstyle = ifelse(is.na(arcstyle), "missing", arcstyle) , 
    gartype = ifelse(is.na(gartype), "missing", gartype),
    attachedGarage = gartype %>% str_to_lower() %>% str_detect("att") %>%as.numeric(),
    detachedGarage = gartype %>% str_to_lower() %>% str_detect("det") %>%as.numeric(),
    carportGarage = gartype %>% str_to_lower() %>% str_detect("cp")%>%as.numeric(),
    noGarage = gartype %>% str_to_lower() %>% str_detect("none")%>%as.numeric()
  ) %>% arrange(parcel,syear,smonth)%>%
  group_by(parcel)%>%
  slice(1) %>%
  ungroup()%>%
  select(-nbhd,-parcel,-status,-qualified,-gartype,-yrbuilt) %>%
  replace_na(
    c(list(
      basement = 0),
      colMeans(select(., nocars, numbdrm, numbaths),
               na.rm = TRUE)
    ) 
  )

dat_mal <- dat %>%
  recipe(before1980 ~ ., data = dat)%>%
  step_dummy(arcstyle) %>%
  prep()%>%
  juice()

write_rds(dat_mal, "dat_mmll.rds")
##data process done 

#explain
library(tidyverse)
library(tidymodels)
library(DALEX)
library(skimr)
library(GGally)
library(xgboost)
library(vip)
library(patchwork)

vis_dat(dat.mmll)

dat_ml <- read_rds("dat_mmll.rds") #follow using explain.r

set.seed(76)
dat_split <- initial_split(dat_ml, prop = 2/3, strata = before1980)

dat_train <- training(dat_split)
dat_test <- testing(dat_split)

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


ggsave(file = "graph1.png", width = 15, height = 9)
(vip(bt_model, num_features = 20) + labs(title = "boosted")) + 
(vip(logistic_model, num_features = 20) + labs(title = "logistic"))


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

##extra
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

ggsave(file = "preds_bt_roccurve.png", width = 15, height = 9)
preds_bt %>%
  roc_curve(truth, estimate = .pred_before)%>%
  autoplot()
dev.off()

ggsave(file ="preds_logistic_roccurve.png",width = 15, height = 9) 
preds_logistic %>%
  roc_curve(truth, estimate = .pred_before)%>%
  autoplot()
dev.off()

ggsave(file ="preds_nb_roccurve.png",width = 15, height = 9) 
preds_nb %>%
  roc_curve(truth, estimate = .pred_before)%>%
  autoplot()
dev.off()

#explain.R
library(tidyverse)
library(tidymodels)
library(DALEX)
library(vip)
library(patchwork)


httpgd::hgd()
httpgd::hgd_browse()

dat_ml <- read_rds("dat_mmll.rds")%>%
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

##DALEX: Model explanation
explainer_bt <- DALEX::explain(
    bt_model,
    select(dat_exp, -before1980), dat_exp$before1980, label = "Boosted Trees")

explainer_logistic <- DALEX::explain(
    logistic_model,
    select(dat_exp, -before1980), dat_exp$before1980, label = "Logistic Regression")

plot(performance_bt, performance_logistic)
plot(performance_bt, performance_logistic, geom = "boxplot")
#looking at the residual area from 0-1, to know the variety of it and the low on the variabjce,
#low in the blue boxplot.

ggsave(file ="reverse_cum.png",width=15, height=9) 
plot(performance_bt, performance_logistic)
dev.off()

ggsave(file ="boxplot.png",width=15, height=9) 
plot(performance_bt, performance_logistic, geom = "boxplot")
dev.off()


logistic_parts <- model_parts(explainer_logistic, 
    loss_function = loss_root_mean_square)
bt_parts <- model_parts(explainer_bt,
    loss_function = loss_root_mean_square)

ggsave(file="loss_function_rtmean.png",width=15, height=9) 
plot(logistic_parts, bt_parts, max_vars = 10)
dev.off()

#DALEX: Explain score
#library(patchwork)
onehouse_before <- predict_parts(explainer_bt,
    new_observation = select(dat_exp, -before1980) %>%
        dplyr::slice(13800), type = "break_down")

onehouse_after <- predict_parts(explainer_bt,
    new_observation = select(dat_exp, -before1980) %>%
        dplyr::slice(8), type = "break_down")

plot(onehouse_after) + plot(onehouse_before)

ggsave(file="break_down.png",width=15, height=9) 
plot(onehouse_after) + plot(onehouse_before)
dev.off()

#Break Down Profile, on left is after and p=.581 and right is p=.283
#dot line avg of prob
# the probability of being a before and after 1980
#waterfall plots- avg of the dataset and shift.
#rig
