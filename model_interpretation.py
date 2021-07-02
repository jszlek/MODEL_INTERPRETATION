import pandas as pd
import shap
import matplotlib.pyplot as plt
# import ipywidgets as widgets
import dalex as dx
from aux_functions import extract_pipeline, prepare_directories
from load_data import load_data


my_data_filename = 'test_h2o_shap.txt'
my_data_sep = '\t'
my_data_output = 'Q'

my_model_filename = 'tpot_best_model.py'
my_model = 'h2o'   # options 'tpot', 'h2o'

my_sample_data = 'kmeans'      # options 'all', 'kmeans'
my_kmeans_n = 1

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
    exported_pipeline.fit(training_features, training_target)
    # init js
    shap.initjs()

    if my_sample_data == 'kmeans':
        # explain sample the predictions in the training set
        explainer = shap.KernelExplainer(exported_pipeline.predict, shap.kmeans(training_features, my_kmeans_n))
    elif my_sample_data == 'all':
        # explain all the predictions in the test set
        explainer = shap.KernelExplainer(exported_pipeline.predict, training_features)

    shap_values = explainer.shap_values(training_features)

    f = plt.figure()
    shap.summary_plot(shap_values, training_features, show=False)
    f.savefig("SHAP_plots/summary_plot.pdf", bbox_inches='tight')
    plt.close()

    f = plt.figure()
    shap.summary_plot(shap_values, training_features, plot_type="bar", show=False)
    f.savefig("SHAP_plots/summary_plot_bar.pdf", bbox_inches='tight')
    plt.close()

    col_list = list(training_features.columns)

    from matplotlib.backends.backend_pdf import PdfPages

    for i in col_list:
        pdf = PdfPages("SHAP_plots/" + i + '_out.pdf')
        shap.dependence_plot(str(i), shap_values, training_features, show=False)
        pdf.savefig()
        pdf.close()

    f = shap.force_plot(explainer.expected_value, shap_values, training_features, plot_cmap="DrDb", show=False)
    shap.save_html("SHAP_html/force_plot.html", f)

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
    h2o_bst_model = h2o.load_model('./StackedEnsemble_AllModels_AutoML_20210702_235201')
    h2o_wrapper = H2OPredWrapper(h2o_bst_model, feature_names)
    # This is the core code for Shapley values calculation

    explainer = shap.KernelExplainer(h2o_wrapper.predict, shap.kmeans(training_features, 2))
    shap_values = explainer.shap_values(training_features)

    # initialize js for SHAP
    shap.initjs()
    f = plt.figure()
    shap.summary_plot(shap_values, training_features, show=False)
    f.savefig("SHAP_plots/summary_plot.pdf", bbox_inches='tight')
    plt.close()

    f = plt.figure()
    shap.summary_plot(shap_values, training_features, plot_type="bar", show=False)
    f.savefig("SHAP_plots/summary_plot_bar.pdf", bbox_inches='tight')
    plt.close()

    col_list = list(training_features.columns)

    from matplotlib.backends.backend_pdf import PdfPages

    for i in col_list:
        pdf = PdfPages("SHAP_plots/" + i + '_out.pdf')
        shap.dependence_plot(str(i), shap_values, training_features, show=False)
        pdf.savefig()
        pdf.close()

    f = shap.force_plot(explainer.expected_value, shap_values, training_features, plot_cmap="DrDb", show=False)
    shap.save_html("SHAP_html/force_plot.html", f)
