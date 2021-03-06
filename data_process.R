
library(tidyverse)
library(tidymodels)
library(visdat)
library(skimr)

httpgd::hgd()
httpgd::hgd_browse()

# Create our target variable (R uses factors not 0/1).
# Handle gartype and arcstyle nominal variables.
# Handle quality and condition ordinal variables.
# Create our garage type varaibles - attachedGarage, detachedGarage, carportGarage, and noGarage.
# Remove duplicated parcels.
# Drop nbhd, parcel, status, qualified, gartype and yrbuilt columns
# Fix columns with missing values.
# One-hot-encode arcstyle.

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
        before1980 = ifelse(yrbuilt < 1980, "before", "after")
            %>% factor(levels = c("before", "after")),
        quality = case_when(
            quality == "E-" ~ -0.3, quality == "E" ~ 0,
            quality == "E+" ~ 0.3, quality == "D-" ~ 0.7, 
            quality == "D" ~ 1, quality == "D+" ~ 1.3,
            quality == "C-" ~ 1.7, quality == "C" ~ 2,
            quality == "C+" ~ 2.3, quality == "B-" ~ 2.7,
            quality == "B" ~ 3, quality == "B+" ~ 3.3,
            quality == "A-" ~ 3.7, quality == "A" ~ 4,
            quality == "A+" ~ 4.3, quality == "X-" ~ 4.7,
            quality == "X" ~ 5, quality == "X+" ~ 5.3),
        condition = case_when(
            condition == "Excel" ~ 3,
            condition == "VGood" ~ 2,
            condition == "Good" ~ 1,
            condition == "AVG" ~ 0,
            condition == "Avg" ~ 0,
            condition == "Fair" ~ -1,
            condition == "Poor" ~ -2),
        arcstyle = ifelse(is.na(arcstyle), "missing", arcstyle),
        gartype = ifelse(is.na(gartype), "missing", gartype),
        attachedGarage = gartype %>% str_to_lower() %>% str_detect("att") %>% as.numeric(),
        detachedGarage = gartype %>% str_to_lower() %>% str_detect("det") %>% as.numeric(),
        carportGarage = gartype %>% str_to_lower() %>% str_detect("cp") %>% as.numeric(),
        noGarage = gartype %>% str_to_lower() %>% str_detect("none") %>% as.numeric(),
    ) %>%
    arrange(parcel, syear, smonth) %>%
    group_by(parcel) %>%
    slice(1) %>%
    ungroup() %>%
    select(-nbhd, -parcel, -status, -qualified, -gartype, -yrbuilt) %>%
    replace_na(
        c(list(
        basement = 0),
        colMeans(select(., nocars, numbdrm, numbaths),
            na.rm = TRUE)
        )
    )

dat_ml <- dat %>%
    recipe(before1980 ~ ., data = dat) %>%
    step_dummy(arcstyle) %>%
    prep() %>%
    juice() #pulls out the data
vis_dat(dat)
glimpse(dat_ml) 
skim(dat_ml) #draws a sum for each variable 

write_rds(dat_ml, "dat_ml.rds")
# arrow::write_feather(dat_ml, "dat_ml.feather")


#Create our garage type varaibles - attachedGarage, detachedGarage, carportGarage, and noGarage.
#Remove duplicated parcels.
vis_dat(dat)


#Let’s clean our data and establish our machine learning data
#We need to do the following;

#Create our target variable (R uses factors not 0/1).
#Handle gartype and arcstyle nominal variables.
#Handle quality and condition ordinal variables.
#Create our garage type varaibles - attachedGarage, detachedGarage, carportGarage, and noGarage.
#Remove duplicated parcels.
#Drop nbhd, parcel, status, qualified, gartype and yrbuilt columns
#Fix columns with missing values.
#One-hot-encode arcstyle.
#As we move through this process let’s leverage visdat::vis_dat()



