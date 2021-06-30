import pandas as pd
import shap
import matplotlib
# import ipywidgets as widgets
import dalex as dx
from extract_tpot_pipeline import extract_pipeline

extract_pipeline(filename='tpot_best_model.py', lines_to_remove=['tpot_data =',
                                                                 'features =',
                                                                 'training_features',
                                                                 'exported_pipeline.fit(',
                                                                 'results =',
                                                                 'train_test_split(features'])

# Load data
tpot_data = pd.read_csv('baza_14in_Cubsit_TPOT_SYMF.txt', sep='\t', dtype=np.float64)
training_features = tpot_data.drop('Q', axis=1)
training_target = tpot_data['Q']

# initialize js for SHAP
shap.initjs()

# explain all the predictions in the test set
explainer = shap.KernelExplainer(exported_pipeline.predict, training_features)

K = 5

# explain sample the predictions in the training set
explainer = shap.KernelExplainer(exported_pipeline.predict, shap.kmeans(training_features, K))

shap_values = explainer.shap_values(training_features)

shap.summary_plot(shap_values, training_features)