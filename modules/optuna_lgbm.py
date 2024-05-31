### 학생때 사용했던 optuna
# 일단 기록

import lightgbm as lgbm
from lightgbm import LGBMClassifier

import optuna

def optuna(X_train,y_train,X_valid,y_valid, model):
  def objective(trial):
    gbm.fit(X_train, y_train, eval_set = [(X_valid,y_valid)], categorical_feature = cat_col, verbose = False, eval_metric = lgbm_aps)

    loss = gbm.best_score_['valid_0']['aps']
    return loss

  sampler = optuna.samplers.TPESampler(seed=100)

  study = optuna.create_study(direction='maximize', sampler=sampler)
  study.optimize(objective, n_trials=100)

  return study.trials_dataframe()

param = { 
    'is_unbalance' : trial.suggest_categorical('is_unbalance',[True,False]),
    'learning_rate' : trial.suggest_loguniform("learning_rate", 0.001, 0.3),
    'reg_alpha' : trial.suggest_loguniform("reg_alpha",1e-8,10.0),
    'reg_lambda' : trial.suggest_loguniform("reg_lambda",1e-8,10.0),
    'num_leaves' : trial.suggest_int('num_leaves',2,256,log=True),
    'colsample_bytree' : trial.suggest_uniform('colsample_bytree',0.4,1.0),
    'subsample' : trial.suggest_uniform('subsample',0.4,1.0)
}

gbm = LGBMClassifier(n_estimators=500, random_state=100, **param)

lgbm_optimized = LGBM_optimize(X_train,y_train,X_valid,y_valid,cat_col, gbm)

lgbm_params = lgbm_optimized.sort_values('value').tail(5).iloc[:,5:-1].apply(lambda x : x.mean(), axis=0)
lgbm_params.index = [x[7:] for x in lgbm_params.index]
lgbm_params = lgbm_params.to_dict()
lgbm_params['num_leaves'] = int(round(lgbm_params['num_leaves']))
lgbm_params['is_unbalance'] = bool(round(lgbm_params['is_unbalance']))