import pandas as pd
import shap
import matplotlib.pyplot as plt
# import ipywidgets as widgets
import dalex as dx
from extract_tpot_pipeline import extract_pipeline
from load_data import load_data

extract_pipeline(filename='tpot_best_model.py', lines_to_remove=['tpot_data =',
                                                                 'features =',
                                                                 'training_features',
                                                                 'exported_pipeline.fit(',
                                                                 'results =',
                                                                 'train_test_split(features'])

training_features, training_target = load_data(filename='baza_14in_Cubsit_TPOT_SYMF.txt', separator='\t', output='Q')

# Load pipeline and fit it to provided data
exec(open('tpot_pipeline.py').read())
exported_pipeline.fit(training_features, training_target)

# init js
shap.initjs()

# explain all the predictions in the test set
explainer = shap.KernelExplainer(exported_pipeline.predict, training_features)

K = 1

# explain sample the predictions in the training set
explainer = shap.KernelExplainer(exported_pipeline.predict, shap.kmeans(training_features, K))

shap_values = explainer.shap_values(training_features)

f = plt.figure()
shap.summary_plot(shap_values, training_features, show=False)
f.savefig("summary_plot.pdf", bbox_inches='tight')
plt.close()

f = plt.figure()
shap.summary_plot(shap_values, training_features, plot_type="bar", show=False)
f.savefig("summary_plot_bar.pdf", bbox_inches='tight')
plt.close()

col_list = list(training_features.columns)

from matplotlib.backends.backend_pdf import PdfPages

for i in col_list:
    pdf = PdfPages(i+'_out.pdf')
    shap.dependence_plot(str(i), shap_values, training_features, show=False)
    pdf.savefig()
    pdf.close()

f=shap.force_plot(explainer.expected_value, shap_values, training_features, plot_cmap="DrDb", show=False)
shap.save_html("force_plot.html", f)
