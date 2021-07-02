import h2o
import pandas as pd
import numpy as np
from h2o.automl import H2OAutoML
from sklearn.model_selection import ShuffleSplit

h2o.init()

data = pd.read_csv('./baza_14in_Cubsit_TPOT_SYMF.txt', sep='\t', engine='python')

ncols = data.shape[1] - 1
nrows = data.shape[0]

X = data.drop(data.columns[[0, ncols]], axis=1)
y = data[data.columns[ncols]]

# split on train - test dataset by group 'Formulation no' - this is for Feature Selection
train_inds, test_inds = next(ShuffleSplit(n_splits=1, train_size=0.7, test_size=0.3).split(X))
X_train, X_test, y_train, y_test = X.iloc[train_inds], X.iloc[test_inds], y.iloc[train_inds], y.iloc[test_inds]

# Write splits on disk
train_set = pd.concat([X_train, y_train], axis=1)
test_set = pd.concat([X_test, y_test], axis=1)

y_idx = train_set.columns[train_set.shape[1] - 1]

training_frame = h2o.H2OFrame(train_set)
testing_frame = h2o.H2OFrame(test_set)

# autoML settings
aml_model = H2OAutoML(max_runtime_secs=90,
                      nfolds=10,
                      keep_cross_validation_models=True,
                      keep_cross_validation_predictions=True,
                      keep_cross_validation_fold_assignment=True,
                      verbosity='info',
                      sort_metric='RMSE')

# train model for FS
aml_model.train(y=y_idx,
                training_frame=training_frame,
                leaderboard_frame=testing_frame)
# saving model
my_model_FS_path = h2o.save_model(aml_model.leader, path='./')