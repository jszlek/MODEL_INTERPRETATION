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
