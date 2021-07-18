# -----------------------------------------------------
# A set of auxilary functions
# -----------------------------------------------------


# -----------------------
# a function to load data
# -----------------------

def load_data(filename: str = None, output: str = None, separator: str = ','):

    valid_sep = {",", "\t", " "}
    if separator not in valid_sep:
        raise ValueError("separator must be one of %r." % valid_sep)

    if filename and output is not None:
        import pandas as pd
        import numpy as np

        tmp_data = pd.read_csv("./"+filename, sep=separator, dtype=np.float64)
        data_input = tmp_data.drop(output, axis=1)
        data_output = tmp_data[output]

    return [data_input, data_output]


# --------------------------------------------------------------
# function to extract only pipeline from exported tpot py script
# --------------------------------------------------------------

def extract_pipeline(filename: str = None, lines_to_remove: list = ['']):
    # check for existance of filename
    if filename is None:
        print("Please provide filename")

    elif filename is not None:
        print("Extracting pipeline from file: ", filename)

        if lines_to_remove != '':
            print("Removing lines starting with:")
            print(lines_to_remove)
            with open(filename) as oldfile, open('tpot_pipeline.py', 'w') as newfile:
                for line in oldfile:
                    if not any(bad_word in line for bad_word in lines_to_remove):
                        newfile.write(line)
            import tpot_pipeline
        elif lines_to_remove == '':
            print("Please provide starting string")
    return 0


# --------------------------------------------------------------------------------
# H2OPredWrapper - class used for obtaining predictions understood by shap package
# --------------------------------------------------------------------------------
class H2OPredWrapper:

    def __init__(self, h2o_model, feature_names):
        self.h2o_model = h2o_model
        self.feature_names = feature_names

    def predict(self, x):
        import h2o
        import pandas as pd

        if isinstance(x, pd.Series):
            x = x.values.reshape(1, -1)
        self.dataframe = pd.DataFrame(x, columns=self.feature_names)
        self.predictions = self.h2o_model.predict(h2o.H2OFrame(self.dataframe)).as_data_frame().values
        return self.predictions.astype('float64')[:, -1]


# --------------------------------------------------------------------------------


# -----------------------
# Prepare dirs
# -----------------------
def prepare_directories():
    from pathlib import Path
    # prepare directory structure
    # get current directory (PosixPath)
    # -----------------------
    my_current_dir = Path.cwd()

    # get export directory and other subdirs (PosixPath)
    # -----------------------
    my_shap_plots_dir = my_current_dir.joinpath(str(my_current_dir) + '/SHAP_plots')
    my_shap_html_dir = my_current_dir.joinpath(str(my_current_dir) + '/SHAP_html')
    my_ale_plots_dir = my_current_dir.joinpath(str(my_current_dir) + '/DALEX_plots')
    my_ale_html_dir = my_current_dir.joinpath(str(my_current_dir) + '/DALEX_html')
    my_h2o_plots_dir = my_current_dir.joinpath(str(my_current_dir) + '/H2O_plots')

    # check subdirectory structure
    # ----------------------------------------
    Path(my_shap_plots_dir).mkdir(parents=True, exist_ok=True)
    Path(my_shap_html_dir).mkdir(parents=True, exist_ok=True)
    Path(my_ale_plots_dir).mkdir(parents=True, exist_ok=True)
    Path(my_ale_html_dir).mkdir(parents=True, exist_ok=True)
    Path(my_h2o_plots_dir).mkdir(parents=True, exist_ok=True)


# -----------------------
# check h2o version
# -----------------------
def check_min_h2o_version():
    import sys
    import h2o
    from packaging import version
    # checking h2o version for use_h2o function
    current_h2o_version = sys.modules[h2o.__package__].__version__
    min_h2o_version = "3.32.1.1"
    print("Your H2O package version is: " + current_h2o_version)
    print("Minimal H2O package version to use this feature is: " + min_h2o_version)
    return version.parse(current_h2o_version) >= version.parse(min_h2o_version)


# ---------------------------
# LIME
# ---------------------------

def use_lime(): # currently unused part of the code
    import lime
    import lime.lime_tabular

    explainer = lime.lime_tabular.LimeTabularExplainer(np.array(X_train), feature_names,
                                                       class_names, categorical_features, mode)

    # np.array(X_train): The training data
    # class_names: The target variable(for regression), different classes in the target
    # variable(
    # for regression)
    # categorical_features: List of all the column names which are categorical
    # mode: For a regression problem: 'regression', and
    # for a classification problem, 'classification


# -----------------------------
# XAI by Dalex
# -----------------------------

def use_dalex(model, data_features, data_target, max_deep_tree, max_vars_tree, explain_preds=None):
    import dalex as dx
    import matplotlib.pyplot as plt
    import plotly
    from sklearn import tree
    from matplotlib.backends.backend_pdf import PdfPages

    my_fitted_model_exp = dx.Explainer(model, data_features, data_target,
                                       label="Partial_dependency_plot")

    # Permutation variable importance
    perm_var_imp = my_fitted_model_exp.model_parts(loss_function='rmse')
    # save data.frame with results
    perm_var_imp.result.to_csv("DALEX_plots/permutation_variable_importance.txt", sep='\t')
    # plot the results
    f = plt.figure()
    perm_var_imp.plot(show=False)
    f.savefig("DALEX_plots/permutation_variable_importance_plot.pdf", bbox_inches='tight')
    plt.close()

    # Partial Dependence Plots
    pd_model = my_fitted_model_exp.model_profile(variables=list(data_features.columns.unique()), type='partial')
    fig = pd_model.plot(show=False)
    plotly.offline.plot(fig, filename='DALEX_plots/pdp_1d_plot.html', auto_open=False)

    surrogate_model = my_fitted_model_exp.model_surrogate(type='tree', max_depth=max_deep_tree, max_vars=max_vars_tree)
    plt.figure(figsize=(16, 9))
    _ = tree.plot_tree(surrogate_model, filled=True, feature_names=data_features.columns, fontsize=9, rounded=True, proportion=True)
    plt.savefig("DALEX_plots/decision_tree_surrogate_model.pdf", bbox_inches='tight')
    plt.close()

    col_list = list(data_features.columns)
    my_model_diagnostics = my_fitted_model_exp.model_diagnostics()

    for i in col_list:
        fig = my_model_diagnostics.plot(variable=i, yvariable='residuals', marker_size=5, show=False)
        plotly.offline.plot(fig, filename=str('DALEX_plots/'+ i +'_residual' +'.html'), auto_open=False)

    if explain_preds is not None:
        for i in range(len(data_features)):
            # Break Down (with predict_parts)
            my_fitted_model_exp_pparts = my_fitted_model_exp.predict_parts(new_observation=data_features.loc[i,], type="break_down")
            # plot Break Down
            fig = my_fitted_model_exp_pparts.plot(show=False)
            plotly.offline.plot(fig, filename=str("DALEX_html/explain_pred_row_no_" + str(i) + '_out.html'), auto_open=False)

    return None


# -----------------------------
# Shapley values by SHAP
# -----------------------------

def use_shap(model, sample_data, kmeans_n, data_features):
    import shap
    import matplotlib.pyplot as plt
    exec(open('tpot_pipeline.py').read())
    # init js
    shap.initjs()

    if sample_data == 'kmeans':
        # explain sample the predictions in the training set
        explainer = shap.KernelExplainer(model.predict, shap.kmeans(data_features, kmeans_n))
    elif sample_data == 'all':
        # explain all the predictions in the test set
        explainer = shap.KernelExplainer(model.predict, data_features)

    shap_values = explainer.shap_values(data_features)

    f = plt.figure()
    shap.summary_plot(shap_values, data_features, show=False)
    f.savefig("SHAP_plots/summary_plot.pdf", bbox_inches='tight')
    plt.close()

    f = plt.figure()
    shap.summary_plot(shap_values, data_features, plot_type="bar", show=False)
    f.savefig("SHAP_plots/summary_plot_bar.pdf", bbox_inches='tight')
    plt.close()

    col_list = list(data_features.columns)

    from matplotlib.backends.backend_pdf import PdfPages

    for i in col_list:
        pdf = PdfPages("SHAP_plots/" + i + '_out.pdf')
        shap.dependence_plot(str(i), shap_values, data_features, show=False)
        pdf.savefig()
        pdf.close()

    f = shap.force_plot(explainer.expected_value, shap_values, data_features, plot_cmap="DrDb", show=False)
    shap.save_html("SHAP_html/force_plot.html", f)

    return None


# ----------------------------------------
# Explain model - internal function of H2O
# ----------------------------------------

def use_h2o(model, data_features, data_target):
    import pandas as pd
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import h2o

    # the data
    dataset = pd.concat([data_features, data_target], axis=1)
    dataset_frame = h2o.H2OFrame(dataset)

    expl_object = model.explain(dataset_frame)

    try:
        pdf = PdfPages("H2O_plots/residuals_h2o_explain_object_output.pdf")
        for i in expl_object.get('residual_analysis').get('plots').items().__iter__():
            pdf.savefig(i[1])
        pdf.close()
    except AttributeError:
        print("Explain object of H2O model has no attribute Residual Analysis")

    try:
        pdf = PdfPages("H2O_plots/pdp_h2o_explain_object_output.pdf")
        for i in expl_object.get('pdp').get('plots').items().__iter__():
            pdf.savefig(i[1])
        pdf.close()
    except AttributeError:
        print("Explain object of H2O model has no attribute Partial Dependence Plot")

    try:
        pdf = PdfPages("H2O_plots/varimp_h2o_explain_object_output.pdf")
        for i in expl_object.get('varimp').get('plots').items().__iter__():
            pdf.savefig(i[1])
        pdf.close()
    except AttributeError:
        print("Explain object of H2O model has no attribute Variable Importance")

    try:
        pdf = PdfPages("H2O_plots/shap_summary_h2o_explain_object_output.pdf")
        for i in expl_object.get('shap_summary').get('plots').items().__iter__():
            pdf.savefig(i[1])
        pdf.close()
    except AttributeError:
        print("Explain object of H2O model has no attribute SHAP summary")

    try:
        pdf = PdfPages("H2O_plots/ice_h2o_explain_object_output.pdf")
        for i in expl_object.get('ice').get('plots').items().__iter__():
            for k in i[1].items():
                pdf.savefig(k[1])
        pdf.close()
    except AttributeError:
        print("Explain object of H2O model has no attribute Individual Conditional Expectation")
