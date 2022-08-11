#Kaggle Competition
library(dplyr)
library(ggplot2)
library(ISLR)
library(caTools)
library(e1071)
library(randomForest)
library(xgboost)
library(Metrics)
library(recipes)
library(tidyverse)
library(forcats)

#Read data
data = read.csv('data/analysisData.csv')
test_data = read.csv('data/scoringData.csv')

#Data Exploration
map(data, ~sum(is.na(.))) #NA values in train data
map(test_data, ~sum(is.na(.))) # NA values in test data 
#drop useless columns with too much text and NA values
#drop columns from train set
data <- subset(data, select= -c(name,
                                summary,
                                space,
                                description,
                                neighborhood_overview,
                                notes,
                                transit,
                                access,
                                interaction,
                                house_rules, 
                                host_name, 
                                host_about, 
                                host_verifications, 
                                host_since,weekly_price,monthly_price,security_deposit,square_feet))

#drop from train set
test_data <- subset(test_data, select= -c(name,
                                          summary,
                                          space,
                                          description,
                                          neighborhood_overview,
                                          notes,
                                          transit,
                                          access,
                                          interaction,
                                          house_rules, 
                                          host_name, 
                                          host_about, 
                                          host_verifications, 
                                          host_since,weekly_price,monthly_price,security_deposit,square_feet))

head(data$host_listings_count)
unique(data$host_listings_count)
unique(data$host_total_listings_count)
unique(data$host_listings_count==data$host_total_listings_count)

#replacing na values in cleaning_fee with mean of it
#train 
data$cleaning_fee[is.na(data$cleaning_fee)] <- mean(data$cleaning_fee, na.rm = TRUE)
data$beds[is.na(data$beds)] <- 0
data$host_listings_count[is.na(data$host_listings_count)] <- 0

#test_data
test_data$cleaning_fee[is.na(test_data$cleaning_fee)] <- mean(test_data$cleaning_fee, na.rm = TRUE)
nrow(data)
test_data$beds[is.na(test_data$beds)] <- 0
test_data$host_total_listings_count[is.na(test_data$host_total_listings_count)] <- 0

#plotting categorical data for outliers
#Histogram for distrubution
dev.off()
ggplot(data = data, aes(x=price))+ geom_histogram(fill='red', binwidth = 5)  
ggplot(data, aes(x=room_type,y=price)) + geom_boxplot() 
ggplot(data, aes(x=host_identity_verified,y=price)) + geom_boxplot()
ggplot(data, aes(x=street,y=price)) + geom_boxplot() 
ggplot(data, aes(x=neighbourhood_group_cleansed,y=price)) + geom_boxplot() 
ggplot(data, aes(x=is_location_exact,y=price)) + geom_boxplot()
ggplot(data, aes(x=instant_bookable,y=price)) + geom_boxplot()
ggplot(data = data, aes(x=guests_included))+ geom_histogram() 
ggplot(data = data, aes(x=property_type, y=price))+ geom_boxplot() 
ggplot(data = both_data, aes(x=number_of_reviews))+ geom_histogram(fill='blue') 
ggplot(data, aes(x=neighbourhood_group_cleansed, y=price)) + geom_boxplot()
ggplot(data, aes(x=room_type, y=price)) + geom_boxplot()
ggplot(data, aes(x=bedrooms)) + geom_bar()
ggplot(data, aes(x=accommodates)) + geom_bar()
ggplot(data, aes(x=price, color=cancellation_policy)) + geom_histogram()
ggplot(data, aes(x=bedrooms, y=price, color=bedrooms)) + geom_point()

#Combining both datasets
test_data$type = 'test'
test_data$price = 0

copy_train_data = data
copy_train_data$type = "train"
combined_set <- rbind.fill(copy_train_data, test_data)

#Function to convert categorical variables to factor
convert_to_factor <- function(main_df, categorical_vars)
{
  
  main_list = categorical_vars
  for (i in 1:length(main_list))
  {
    new_var = main_list[i]
    main_df[,new_var] <- as.factor(main_df[,new_var])
    
  }
  
  return (main_df)
  
}

#Creating numeric column out of host_response_rate
combined_set$host_response_rate <- as.numeric(sub("%","",combined_set$host_response_rate))/100

#Replacing missing values with median of columns
fresh_set = combined_set %>%
  mutate_if(is.numeric, funs(replace(., is.na(.), mean(., na.rm = TRUE))))

library(readr)
tmp <- as.character(parse_number(fresh_set$zipcode))
#length(unique(tmp))

#removing text from zipcode
fresh_set$zipcode <- gsub("[^\\d]", "", fresh_set$zipcode, perl=TRUE)
# zip code digits from 1 to 5
fresh_set$zipcode <- substr(fresh_set$zipcode, 1, 5)
fresh_set$zipcode[nchar(fresh_set$zipcode)<5] <- NA_character_
fresh_set$zipcode <- as.factor(fresh_set$zipcode)
fresh_set$new_zip_code <- fct_lump_n(fresh_set$zipcode, 50)

#Converting variables to factor
main_categorical = colnames(fresh_set[, sapply(fresh_set, class) == 'character'])


categorical_vars = c(
  "cancellation_policy",                                                                
  "is_business_travel_ready",                                                                    
  "host_response_time",                                                                      
  "instant_bookable",
  "new_zip_code" ,
  "review_scores_cleanliness",
  "property_type"
)


#Converting all categorical variables to factor variables
factored_df = convert_to_factor(fresh_set,main_categorical)

#Split entire data to train and test

master_train_df = factored_df[(factored_df$type == 'train'),]
master_test_df = factored_df[(factored_df$type == 'test'),]

#Code for generating results on the entire dataset
#Running the model on the entire dataset and predicting for scoring data

master_train_df= subset(master_train_df, select = -c(type))
master_test_df= subset(master_test_df, select = -c(type))

#model
variables_to_use = "accommodates+bathrooms+bedrooms+guests_included+extra_people+minimum_nights+maximum_nights+minimum_minimum_nights+maximum_minimum_nights+minimum_maximum_nights+calculated_host_listings_count_shared_rooms+calculated_host_listings_count_private_rooms+calculated_host_listings_count+maximum_maximum_nights+minimum_nights_avg_ntm+maximum_nights_avg_ntm+host_response_rate+host_location+host_response_time+host_is_superhost+host_neighbourhood+host_has_profile_pic+host_identity_verified+street+neighbourhood+neighbourhood_cleansed+neighbourhood_group_cleansed+city+state+market+smart_location+country_code+country+is_location_exact+property_type+room_type+bed_type+amenities+calendar_updated+has_availability+first_review+last_review+requires_license+license+jurisdiction_names+instant_bookable+is_business_travel_ready+cancellation_policy+require_guest_profile_picture+require_guest_phone_verification+number_of_reviews+new_zip_code"
temp = unlist(strsplit(variables_to_use, "\\+"))

train_df = master_train_df[, c(temp,"price")]
val_df = master_test_df[, c(temp)]

train_x = data.matrix(train_df[, names(train_df) != "price"])
train_y = train_df[,'price']

test_x = data.matrix(val_df)

xgb_train = xgb.DMatrix(data = train_x, label = train_y)
xgb_test = xgb.DMatrix(data = test_x)

params <- list(booster = "gbtree", objective = "reg:squarederror", eta=0.15, gamma=0, max_depth=7, min_child_weight=1, subsample=1, colsample_bytree=1)
xgb_cv <- xgb.cv(params = params, data = xgb_train, nrounds = 150, nfold = 5, showsd = T, stratified = T, print_every_n = 10, early_stopping_rounds = 20, maximize = F)
xgbc <- xgb.train (params = params, data = xgb_train, nrounds = xgb_cv$best_iteration, watchlist = list(train=xgb_train), print_every_n = 10, early_stopping_rounds = 10, maximize = F , eval_metric = "rmse")
pred_y = predict(xgbc, xgb_test)

#Generating the test file
submissionFile = data.frame(id = test_data$id, price = pred_y)
write.csv(submissionFile, paste0(format(Sys.time(), "%d_%b_%H_%M"),"submission", ".csv"), row.names = F)


