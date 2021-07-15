import pandas as pd
import shap
import matplotlib.pyplot as plt
# import ipywidgets as widgets
import dalex as dx
from aux_functions import extract_pipeline, prepare_directories, use_shap, use_dalex, use_h2o, check_min_h2o_version
from load_data import load_data


my_data_filename = 'test_h2o_shap.txt'
my_data_sep = '\t'
my_data_output = 'Q'

my_model_filename = 'StackedEnsemble_AllModels_AutoML_20210707_132949'
my_model = 'h2o'   # options 'tpot', 'h2o'

my_sample_data: str = 'kmeans'      # options 'all', 'kmeans'
my_kmeans_n = 2

# methods

if_use_shap = False
if_use_pdp = True
if_use_ale = True
if_use_iml = True
if_use_dalex = False
if_use_eli5 = True
if_use_lime = True
if_use_h2o = True   # can be applied only for h2o models and H2O platform



if my_model == 'tpot':

    prepare_directories()

    training_features, training_target = load_data(filename=my_data_filename, separator=my_data_sep, output=my_data_output)

    extract_pipeline(filename=my_model_filename, lines_to_remove=['tpot_data =',
                                                                 'features =',
                                                                 'training_features',
                                                                 'exported_pipeline.fit(',
                                                                 'results =',
                                                                 'train_test_split(features'])

    # Load pipeline and fit it to provided data
    exec(open('tpot_pipeline.py').read())
    my_fitted_model = exported_pipeline.fit(training_features, training_target)

    if if_use_dalex == True:
        use_dalex(model=my_fitted_model, data_features=training_features,
                  data_target=training_target, max_deep_tree=5,
                  max_vars_tree=5, explain_preds=training_features)

    if if_use_shap == True:
        use_shap(model=my_fitted_model, data_features=training_features,
                 sample_data=my_sample_data, kmeans_n=my_kmeans_n)



if my_model == 'h2o':

    import h2o
    from aux_functions import H2OPredWrapper
    from h2o.automl import H2OAutoML

    prepare_directories()

    # -------------------------------------
    # run h2o server
    # -------------------------------------
    h2o.init()
    h2o.no_progress()

    training_features, training_target = load_data(filename=my_data_filename, separator=my_data_sep,
                                                   output=my_data_output)

    feature_names = list(training_features.columns)

    # We will load saved model rather than extract the best model
    # h2o_bst_model = aml_10cv.leader
    h2o_bst_model = h2o.load_model('./StackedEnsemble_AllModels_AutoML_20210707_132949')
    h2o_wrapper = H2OPredWrapper(h2o_bst_model, feature_names)
    # This is the core code for Shapley values calculation

    if if_use_shap == True:
        use_shap(h2o_wrapper, my_sample_data, my_kmeans_n, training_features)

    if if_use_dalex == True:
        use_dalex(model=h2o_wrapper, data_features=training_features, data_target=training_target, max_deep_tree=5, max_vars_tree=5)

    if if_use_h2o == True:

        if check_min_h2o_version():
            use_h2o()
        else:
            print("Please update H2O package! For anaconda users in console type:"+
                  "\n" +
                  "'conda install -c h2oai -y h2o'>=3.32.1.1''")


