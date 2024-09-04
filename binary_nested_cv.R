# Load libraries ----------------------------------------------------------
library(tidymodels)
library(forcats)
library(themis)
#library(tictoc)
library(doFuture)
library(discrim)

#Conflicts
conflicted::conflict_prefer("spec", "yardstick")
conflicted::conflicts_prefer(dplyr::filter)
# Read files --------------------------------------------------------------
ml_dat <- read.csv("binary_na_rm_af_repl_04-02.csv") %>%
  mutate_at(c("Assertion", "Func.refGene", "ExonicFunc.refGene"), as.factor) %>%
  mutate(Assertion = fct_relevel(Assertion, "P_LP")) 


# Split data --------------------------------------------------------------

# Case weights
class_weights <- ml_dat %>%
  group_by(Func.refGene, Assertion) %>%
  summarise(n = n(), .groups = 'drop') 


# Calculate frequencies
freqs <- ml_dat %>%
  group_by(Func.refGene, Assertion) %>%
  tally(name = "count") %>%
  mutate(freq = count / sum(count))

# Calculate importance weights as the inverse of frequencies
weights <- freqs %>%
  mutate(weights = importance_weights(1 / freq)) %>%
  select(Func.refGene, Assertion, weights)

# Join weights back to the main dataset
ml_dat <- ml_dat %>%
  left_join(weights, by = c("Func.refGene", "Assertion"))

set.seed(9815)
split <- initial_split(ml_dat,
                       prop = 0.7)
train_dat <- training(split)
test_dat <- testing(split)



folds <- nested_cv(train_dat,
                   outside = vfold_cv(v = 10, repeats = 5),
                   inside = bootstraps(times = 25))



# Model spec --------------------------------------------------------------


lr_spec <- logistic_reg(penalty = tune(), 
                        mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

nb_spec <- naive_Bayes(smoothness = tune(),
                       Laplace = tune()) %>%
  set_engine("naivebayes") %>%
  set_mode("classification")

rf_spec <- rand_forest(mtry = tune(),
                       trees = tune(),
                       min_n = tune()) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")


lda_spec <- discrim_linear(penalty = tune()) %>%
  set_engine("mda") %>%
  set_mode("classification")

xgb_spec <- 
  boost_tree(mtry = tune(), 
             trees = tune(), 
             min_n = tune(), 
             tree_depth = tune(), 
             learn_rate = tune(), 
             loss_reduction = tune(), 
             sample_size = tune()) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification") 

knn_spec <- 
  nearest_neighbor(neighbors = tune(), 
                   weight_func = tune(), 
                   dist_power = tune()) %>% 
  set_engine("kknn") %>% 
  set_mode("classification")

svm_p_spec <- 
  svm_poly(cost = tune(), degree = tune()) %>% 
  set_engine("kernlab") %>% 
  set_mode("classification")

nnet_spec <- 
  mlp(hidden_units = tune(), penalty = tune(), epochs = tune()) %>% 
  set_engine("nnet", MaxNWts = 2600) %>% 
  set_mode("classification")


mod_spec_ls <- list(rf_spec,
                    xgb_spec,
                    lr_spec,
                    lda_spec,
                    nnet_spec,
                    knn_spec,
                    svm_p_spec)


# Recipes -----------------------------------------------------------------


base_rec <- recipe(Assertion ~ .,
                   data = train_dat) %>%
  update_role(ID, new_role = "ID") %>%
  step_impute_bag(all_numeric_predictors()) %>%
  step_impute_knn(all_nominal_predictors()) %>%
  step_novel(all_nominal_predictors()) %>%
  step_zv(all_numeric_predictors()) %>%
  step_downsample(Assertion)


decorr_rec <- base_rec  %>%
  step_corr(all_numeric_predictors(),
            threshold = 0.7)

dummy_decorr_rec <- base_rec %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors(),
            threshold = 0.7)

dummy_zv_decorr_trans_rec <- base_rec %>%
  step_YeoJohnson(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors(),
            threshold = 0.7)

all_rec <- dummy_zv_decorr_trans_rec %>%
  step_normalize()

rec_ls <- list(decorr_rec,
               dummy_decorr_rec,
               dummy_zv_decorr_trans_rec,
               dummy_zv_decorr_trans_rec,
               all_rec,
               all_rec,
               all_rec)


test <- decorr_rec %>%
  prep() %>%
  bake(new_data = NULL) 

# Workflows ---------------------------------------------------------------


rf_wflow <- workflow_set(
  preproc = list(decorr = decorr_rec),
  models = list(RF = rf_spec),
  case_weights = weights
)

#While tree-based boosting methods generally do not require the creation of dummy variables, models using the xgboost engine do 
xgb_wflow <- workflow_set(
  preproc = list(dummy_decorr = dummy_decorr_rec),
  models = list(XGB = xgb_spec),
  case_weights = weights
)

dummy_zv_decorr_trans_wflow <- workflow_set(
  preproc = list(dummy_zv_decorr_trans = dummy_zv_decorr_trans_rec),
  models = list(LR = lr_spec,
                LDA = lda_spec),
  case_weights = weights
)

all_preproc_wflow <- workflow_set(
  preproc = list(all_preproc = all_rec),
  models = list(NNET = nnet_spec,
                KNN = knn_spec,
                SVM = svm_p_spec))

all_workflows <- bind_rows(rf_wflow,
                           xgb_wflow,
                           dummy_zv_decorr_trans_wflow,
                           all_preproc_wflow
)


rf_param <- extract_parameter_set_dials(rf_spec) %>%
  finalize(x = train_dat %>% 
             select(-Assertion))

xgb_param <- extract_parameter_set_dials(xgb_spec) %>%
  finalize(x = train_dat %>% 
             select(-Assertion))

all_workflows <- all_workflows %>%
  option_add(param_info = rf_param,
             id = "decorr_RF")

all_workflows <- all_workflows %>%
  option_add(param_info = xgb_param,
             id = "dummy_decorr_XGB")



# Tuning ------------------------------------------------------------------

bayes_ctrl <- control_bayes(
  verbose = TRUE,
  no_improve = 10,
  save_pred = TRUE,
  parallel_over = "everything"
)

#Tune hyperparameters across inner folds
tune_inner <- function(inner_rs, wflow_set){
  wflow_set %>% workflow_map(seed = 26503,
                             fn = "tune_bayes",
                             resamples = inner_rs,
                             metrics = metric_set(roc_auc, recall, precision, f_meas, 
                                                  accuracy, kap, sens, spec),
                             objective = exp_improve(),
                             iter = 50, 
                             control = bayes_ctrl,
                             initial = 10,
                             verbose = TRUE)
}

#tic()
set.seed(111)
doFuture::registerDoFuture()
tune_results <- map(folds$inner_resamples, 
                    tune_inner, 
                    all_workflows)
#toc()


get_hyperparams <- function(tuning_results, model){
  tuning_results %>%
    filter(grepl(as.character(model), wflow_id)) %>%
    mutate(metrics = map(result, collect_metrics)) %>%
    select(wflow_id, metrics) %>%
    unnest(cols = metrics)
}

mod_list <- list(RF = "decorr_RF", 
                 XGB = "dummy_decorr_XGB",
                 LR = "dummy_zv_decorr_trans_LR",
                 LDA = "dummy_zv_decorr_trans_LDA",
                 NNET = "all_preproc_NNET",
                 KNN = "all_preproc_KNN",
                 SVM = "all_preproc_SVM")

#apply hp function to each of the different models/algorithms 
hp_list <- foreach(i = mod_list) %dofuture% {
  hp_ls <- map(tune_results, get_hyperparams, i)
  return(hp_ls)
}

#tidy up names to match mod_list
names(hp_list) <- names(mod_list)

best_roc <- function(dat) dat[which.max(dat$mean),]

#determine best param estimate for each of the outer resampling iterations
roc_vals <- foreach(i = 1:length(hp_list)) %dofuture% {
  hp_list[[i]] %>%
    map_df(best_roc) %>%
    select(-c(".metric",
              ".estimator",
              # "mean",
              "n",
              "std_err",
              ".config",
              ".iter",
              "wflow_id"))
}


names(roc_vals) <- names(mod_list)

#Bind params to folds
rs <- foreach(i=1:length(roc_vals))%dofuture% {
  bind_cols(roc_vals[[i]],
            folds)
}

names(rs) <- names(mod_list)

#Nest param info
rs$RF <- rs$RF %>%
  nest(params = c(
    mtry,
    trees,
    min_n))

rs$XGB <- rs$XGB %>%
  nest(params = c(
    mtry,
    trees,
    min_n,
    tree_depth,
    loss_reduction,
    sample_size,
    learn_rate
  ))

rs$LR <- rs$LR %>%
  nest(params = c(
    penalty,
    mixture))

rs$LDA <- rs$LDA %>%
  nest(params = c(
    penalty))

rs$NNET <- rs$NNET %>%
  nest(params = c(
    hidden_units,
    penalty,
    epochs))

rs$KNN <- rs$KNN %>%
  nest(params = c(
    neighbors,
    weight_func,
    dist_power))

rs$SVM <- rs$SVM %>%
  nest(params = c(
    cost,
    degree))

update_mods <- function(mod_params, mod_spec) {
  foreach(i = 1:nrow(mod_params)) %dofuture% {
    #there are issues with LDA, workaround to ensure penalty value is evaluated properly
    if (mod_spec$engine == "mda") {
      penalty_value <- mod_params$params[[i]] %>% pull(penalty)
      mod_spec %>% update(penalty = !!penalty_value)
    } else {
    mod_spec %>%
      update(mod_params$params[[i]])
    }
  }
}

#Update mod specs with best hp combos 

updated_mods <- map2(rs,
                     mod_spec_ls,
                     update_mods)

#fit outer folds with best hyperparameters from inner tuning
fit_outer <- function(object, model_upd, rec, modname){
  
  #make sure all params are quosures
  model_upd$args <- lapply(model_upd$args, quo)
  
  #update workflow with hyperparams & rec, fit to training set
  upd_wf <- workflow() %>%
    add_model(model_upd) %>%
    add_recipe(rec) %>%
    fit(analysis(object))
  
  #predict on validation set
  holdout_pred <- predict(upd_wf,
                          assessment(object) %>%
                            select(-Assertion),
                          type = "prob") %>%
    bind_cols(assessment(object) %>% 
                select(Assertion)) 
  return(holdout_pred)
  
}


outer_rs <- foreach(i = 1:length(rs)) %dofuture% {
  map2(rs[[i]]$splits,
       updated_mods[[i]],
       fit_outer,
       rec_ls[[i]])
}

names(outer_rs) <- names(mod_list)

#Get hard class probabilities
hard_preds <- function(preds){
  foreach(i = 1:length(preds)) %dofuture% {
  preds[[i]] %>% 
    mutate(pred_class = if_else(.pred_P_LP > .pred_B_LB, "P_LP", "B_LB")) %>% 
      mutate(pred_class = as.factor(pred_class)) %>% 
      mutate(pred_class = fct_relevel(pred_class, "P_LP")) %>% 
      mutate(resample = paste0("fold", i))
  }
}


rs_hard_preds <- map(outer_rs, hard_preds)

rs_hard_preds <- map(rs_hard_preds, bind_rows)

rs_preds_names <- foreach(i = 1:length(rs_hard_preds)) %dofuture% {
  rs_hard_preds[[i]] %>% 
    mutate(model = names(rs_hard_preds)[[i]])
}
names(rs_preds_names) <- names(mod_list)

#Define metrics
multi_metric <- metric_set(recall, precision, f_meas, accuracy,
                           roc_auc, sens, spec)

#Get metrics for each model
get_cv_metrics <- function(preds){
   preds$pred_class <- factor(preds$pred_class, levels = c("P_LP", "B_LB"))
  preds %>% 
    group_by(resample) %>%
    multi_metric(truth = Assertion, 
                 .pred_P_LP,
                 estimate = pred_class) 
}

cv_metrics <- map(rs_preds_names, get_cv_metrics)


#Get average metrics across all folds
get_ave_metrics <- function(metrics){
  metrics %>% 
    group_by(.metric) %>% 
    summarise(ave_metric = mean(.estimate))
}

ave_metrics <- map(cv_metrics, get_ave_metrics)
#RF shows best performance, move forward with RF

#roc curves
merged_preds <- rs_preds_names %>% 
  bind_rows()

merged_preds %>% 
  group_by(model) %>% 
  roc_curve(truth = Assertion,
            .pred_P_LP) %>%
  autoplot() +
theme(axis.text = element_text(size=10),
      axis.title=element_text(size=12),
      strip.text = element_text(size = 10),
      legend.title = element_text(size=12),
      legend.text = element_text(size = 10)) 

#Determine best hp
#Bind roc_auc to hp
rf_hp_roc <- cv_metrics$RF %>% 
  filter(.metric == "roc_auc") %>%
  mutate(resample = factor(resample, levels = c(paste0("fold", 1:9), "fold10"))) %>%
  arrange(resample) %>% 
  bind_cols(rs$RF) %>%
  unnest(params)


#Choose best hyperparams for best model - those that were most often chosen to be used on outer folds
best_hp <- rf_hp_roc %>%
  group_by(mtry, trees, min_n) %>%
  tally() %>%
  ungroup() %>% 
  filter(n == max(n)) %>%
  select(-n)
#If no repeats choose one with highest roc_auc
best_hp <- rf_hp_roc %>% 
  filter(.metric == "roc_auc") %>%
  arrange(desc(.estimate)) %>%
  dplyr::slice(1) %>% 
  select(mtry, trees, min_n)

# Evaluate on test --------------------------------------------------------

#update model spec
rf_u_spec <- rf_spec %>%
  update(best_hp)

#Oversample recipe
over_rec <- recipe(Assertion ~ .,
                   data = train_dat) %>%
  update_role(ID, new_role = "ID") %>%
  step_impute_bag(all_numeric_predictors()) %>%
  step_impute_knn(all_nominal_predictors()) %>%
  step_novel(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_numeric_predictors()) %>%
  step_smote(Assertion) 

over_bake <- over_rec %>%
  prep() %>%
  bake(new_data = NULL)


final_wf <- workflow() %>%
  add_recipe(over_rec) %>%
  add_model(rf_u_spec)

set.seed(123)
final_rs <- final_wf %>%
  last_fit(split,
           metrics = metric_set(recall, precision, f_meas, 
                                accuracy, kap,
                                roc_auc, sens, spec))


final_rs %>%
  collect_metrics()

final_rs %>% 
  collect_predictions() %>% 
  conf_mat(truth = Assertion,
           estimate = .pred_class) %>% 
  autoplot(type = "heatmap") +
  theme(axis.text = element_text(size=10),
        axis.title=element_text(size=12),
        strip.text = element_text(size = 10)) 

final_rf_wflow <- final_rs %>%
  extract_workflow()

final_rf_wflow %>% 
  extract_fit_parsnip() %>%
  vip(num_features = 50)  +
  theme(axis.text = element_text(size=10),
        axis.title=element_text(size=12),
        strip.text = element_text(size = 10)) 

#Save trained model
saveRDS(final_rf_wflow,
        "weighted_rf2.rds")