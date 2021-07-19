import pandas as pd
import shap
import matplotlib.pyplot as plt
import dalex as dx
from aux_functions import extract_pipeline, prepare_directories, use_shap, use_dalex, use_h2o, check_min_h2o_version, load_data
from configparser import ConfigParser

# config parser init
config = ConfigParser(allow_no_value=True)
config.read('config.ini')

# data config
my_data_filename = config['DEFAULT']['my_data_filename']
my_data_sep = config['DEFAULT']['my_data_sep']
my_data_output = config['DEFAULT']['my_data_output']

# explain in

# model filename and type
my_model_filename = config['DEFAULT']['my_model_filename']
my_model = config['DEFAULT']['my_model']

# sampling data config
my_sample_data = config['DEFAULT']['my_sample_data']
my_kmeans_n = config['DEFAULT'].getint('my_kmeans_n')

# methods used config
if_use_shap = config['DEFAULT'].getboolean('if_use_shap')
if_use_dalex = config['DEFAULT'].getboolean('if_use_dalex')
if_use_h2o = config['DEFAULT'].getboolean('if_use_h2o')

# surrogate model config
my_max_deep_tree = config['DEFAULT'].getint('my_max_deep_tree')
my_max_vars_tree = config['DEFAULT'].getint('my_max_vars_tree')

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
                  data_target=training_target, max_deep_tree=my_max_deep_tree,
                  max_vars_tree=my_max_vars_tree, explain_preds=training_features)

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
    h2o_bst_model = h2o.load_model(my_model_filename)
    h2o_wrapper = H2OPredWrapper(h2o_bst_model, feature_names)
    # This is the core code for Shapley values calculation

    if if_use_shap == True:
        use_shap(h2o_wrapper, my_sample_data, my_kmeans_n, training_features)

    if if_use_dalex == True:
        use_dalex(model=h2o_wrapper, data_features=training_features, data_target=training_target, max_deep_tree=5, max_vars_tree=5)

    if if_use_h2o == True:

        if check_min_h2o_version():
            use_h2o(model=h2o_bst_model, data_features=training_features, data_target=training_target)
        else:
            print("Please update H2O package! For anaconda users in console type:"+
                  "\n" +
                  "'conda install -c h2oai -y h2o'>=3.32.1.1''")


