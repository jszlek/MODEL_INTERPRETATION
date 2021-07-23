# MODEL_INTERPRETATION
A model interpretation repo.  
The script interprets tpot and h2o models.  
It has shap, dalex and internal h2o interpretation methods implemented.  

Please modify `config.ini` and start `model_interpretation.py` script.

Required packages:
`h2o`
`tpot`
`shap`
`dalex`
`pandas`
`statsmodels`
`numpy`
`scipy`
`scikit-learn`
`matplotlib`
`ipython`
`xgboost`

Proposed mode of installation in conda (should work for ptyhon>=3.6.5):  
`conda create -n for_h2o`  
`conda install -c h2oai h2o`  
`conda install tpot shap dalex pandas statsmodels numpy scipy scikit-learn matplotlib ipython xgboost`  
