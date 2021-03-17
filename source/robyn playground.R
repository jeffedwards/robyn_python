# https://facebookexperimental.github.io/Robyn/
# https://facebookexperimental.github.io/Robyn/docs/step-by-step-guide
# install.packages("glmnet", dependencies=TRUE)

########################################################################################################################
# LOAD PACKAGES
library(glmnet)
library(reticulate)
require(stringr)
require(data.table)
require(minpack.lm)
require(ggplot2)
require(ggpubr)
require(gridExtra)
require(prophet)
require(doParallel)
require(rPref)
require(gridExtra)



########################################################################################################################
# SETUP CONDA ENVIRONMENT
# conda_create("r-reticulate") # must run this line once only
# conda update -n base -c defaults conda  # run in terminal
# conda_install("r-reticulate", "nevergrad", pip=TRUE) # must install nevergrad in conda before running Robyn
use_condaenv("r-reticulate")


########################################################################################################################
# LOAD DATA
script_path <- str_sub(rstudioapi::getActiveDocumentContext()$path, start = 1, end = max(unlist(str_locate_all(rstudioapi::getActiveDocumentContext()$path, "/"))))
dt_input <- fread(paste0(script_path, 'de_simulated_data.csv')) # input time series should be daily, weekly or monthly
dt_holidays <- fread(paste0(script_path, 'holidays.csv')) # when using own holidays, please keep the header c("ds", "holiday", "country", "year")

# source(paste(script_path, "fb_robyn.func.R", sep=""))
# source(paste(script_path, "fb_robyn.optm.R", sep=""))
source(paste0(script_path, "fb_robyn.func.R"))
source(paste0(script_path, "fb_robyn.optm.R"))


########################################################################################################################
# SET MODEL INPUT VARIABLES
set_country <- "DE" # only one country allowed. Used in prophet holidays

# VARIABLE NAMES
set_dateVarName <- "DATE" # date must be format "2020-01-01"
set_depVarName <- "revenue" # there should be only one dependent variable
# set_depVarType <- "revenue" # "revenue" or "conversion" are allowed

# PROPHET
activate_prophet <- T
set_prophet <- c("trend", "season", "holiday")
set_prophetVarSign <- c("default","default", "default")

# BASE VARIABLES
# activate_baseline <- T
set_baseVarName <- "competitor_sales_B"
# set_baseVarName <- c("promotions", "price changes", "competitors sales")
set_baseVarSign <- "negative"
# set_baseVarSign <- c("negative","default","negative") #“positive” is the remaining option

# MEDIA
set_mediaVarName <- c("tv_S"    ,"ooh_S",   "print_S"   ,"facebook_I"   ,"search_clicks_P")
# set_mediaSpendName <- c("tv_S"  ,"ooh_S",   "print_S"   ,"facebook_S"   ,"search_S")
set_mediaVarSign <- c("positive", "positive", "positive", "positive", "positive")
# set_factorVarName <- c()

# CORES FOR PARALLEL PROCESSING
library(foreach)
registerDoSEQ()
library(parallel)
detectCores()
set_cores <- 16


# MODEL TRAINING SIZE
f.plotTrainSize(F) # insert TRUE to plot training size guidance.
set_modTrainSize <- 0.74


# MODEL CORE FEATURES
adstock <- "geometric"
set_iter <- 500
# set_iter <- 20 # Monkey
set_hyperOptimAlgo <- "DiscreteOnePlusOne"
set_trial <- 40
# set_trial <- 2 # Monkey


# HYPERPARAMETER BOUNDS
#### set hyperparameters

set_hyperBoundLocal <- list(
  facebook_I_alphas = c(0.5, 3) # example bounds for alpha
 ,facebook_I_gammas = c(0.3, 1) # example bounds for gamma
 ,facebook_I_thetas = c(0, 0.3) # example bounds for theta
 #,facebook_I_shapes = c(0.0001, 2) # example bounds for shape
 #,facebook_I_scales = c(0, 0.1) # example bounds for scale

  ,ooh_S_alphas = c(0.5, 3)
  ,ooh_S_gammas = c(0.3, 1)
  ,ooh_S_thetas = c(0.1, 0.4)
 #,ooh_S_shapes = c(0.0001, 2)
 #,ooh_S_scales = c(0, 0.1)

  ,print_S_alphas = c(0.5, 3)
  ,print_S_gammas = c(0.3, 1)
 ,print_S_thetas = c(0.1, 0.4)
 #,print_S_shapes = c(0.0001, 2)
 #,print_S_scales = c(0, 0.1)

  ,tv_S_alphas = c(0.5, 3)
  ,tv_S_gammas = c(0.3, 1)
  ,tv_S_thetas = c(0.3, 0.8)
 #,tv_S_shapes = c(0.0001, 2)
 #,tv_S_scales= c(0, 0.1)

  ,search_clicks_P_alphas = c(0.5, 3)
  ,search_clicks_P_gammas = c(0.3, 1)
  ,search_clicks_P_thetas = c(0, 0.3)
 #,search_clicks_P_shapes = c(0.0001, 2)
 #,search_clicks_P_scales = c(0, 0.1)
)


# DEFINE GROUND-TRUTH CALIBRATION
#### define ground truth
activate_calibration <- F # Switch to TRUE to calibrate model.
set_lift <- data.table(channel = c("facebook_I",  "tv_S", "facebook_I"),
                       liftStartDate = as.Date(c("2018-05-01", "2017-11-27", "2018-07-01")),
                       liftEndDate = as.Date(c("2018-06-10", "2017-12-03", "2018-07-20")),
                       liftAbs = c(400000, 300000, 200000))


# should update and remove
local_name <- f.getHyperNames()  # added since I was getting an error; 2021.03.17

# PREPARE THE INPUT DATA
dt_mod <- f.inputWrangling()


# RUN MODELS
model_output_collect <- f.robyn(set_hyperBoundLocal
                                ,optimizer_name = set_hyperOptimAlgo
                                ,set_trial = set_trial
                                ,set_cores = set_cores
                                ,plot_folder = "~/Documents/GitHub/plots")


