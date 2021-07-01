
def plot_figs(shap_values, training_features):
    import matplotlib.pyplot as plt
    import shap

    f = plt.figure()
    shap.summary_plot(shap_values, training_features)
    f.savefig("summary_plot.pdf", bbox_inches='tight')