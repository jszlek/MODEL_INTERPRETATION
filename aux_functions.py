# function to extract only pipeline from exported tpot py script


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
    my_ale_plots_dir = my_current_dir.joinpath(str(my_current_dir) + '/ALE_plots')

    # check subdirectory structure
    # ----------------------------------------
    Path(my_shap_plots_dir).mkdir(parents=True, exist_ok=True)
    Path(my_shap_html_dir).mkdir(parents=True, exist_ok=True)
    Path(my_ale_plots_dir).mkdir(parents=True, exist_ok=True)


# ---------------------------
# LIME
# ---------------------------

def use_lime():
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
# PDP
# -----------------------------

def use_pdp(model, data_features, data_target):
    import dalex as dx
    import plotly

    my_fitted_model_exp = dx.Explainer(model, data_features, data_target,
                                       label="Partial_dependency_plot")
    pd_model = my_fitted_model_exp.model_profile(variables=list(data_features.columns.unique()))
    fig = pd_model.plot(show=False)
    plotly.offline.plot(fig, filename='SHAP_plots/pdp_1d_plot.html', auto_open=False)


# -----------------------------
# SHAP
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
