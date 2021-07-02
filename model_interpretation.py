import pandas as pd
import shap
import matplotlib.pyplot as plt
# import ipywidgets as widgets
import dalex as dx
from extract_tpot_pipeline import extract_pipeline
from load_data import load_data
from pathlib import Path

my_data_filename = './baza_14in_Cubsit_TPOT_SYMF.txt'
my_data_sep = '\t'
my_data_output = 'Q'

my_model_filename = 'tpot_best_model.py'
my_model = 'tpot'   # options 'tpot', 'h2o'

my_sample_data = 'kmeans'      # options 'all', 'kmeans'
my_kmeans_n = 1

if my_model == 'tpot':
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

    # prepare directory structure
    # get current directory (PosixPath)
    # -----------------------
    my_current_dir = Path.cwd()

    # get export directory and other subdirs (PosixPath)
    # -----------------------
    my_shap_plots_dir = my_current_dir.joinpath(str(my_current_dir) + '/SHAP_plots')
    my_shap_html_dir = my_current_dir.joinpath(str(my_current_dir) + '/SHAP_html')
    my_ale_plots_dir = my_current_dir.joinpath(str(my_current_dir) + '/ALE_plots')

    # check subdirectory structure
    # ----------------------------------------
    Path(my_shap_plots_dir).mkdir(parents=True, exist_ok=True)
    Path(my_shap_html_dir).mkdir(parents=True, exist_ok=True)
    Path(my_ale_plots_dir).mkdir(parents=True, exist_ok=True)

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
