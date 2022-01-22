# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ---------------------------------- IMPORTING LIBRARIES --------------------------------
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# ------------------------------- System defined Packages -------------------------------
from flask import Flask, request, render_template, url_for
import numpy as np
import pandas as pd
from werkzeug.datastructures import FileStorage



# -------------------------------- User defined Packages --------------------------------
from GlobalVariables.GlobalVariables import GlobalVariablesPath
from Data_Preprocessing.CleaningData import Cleaning
from Data_Preprocessing.SelectingFeatures import FeatureSelection
from Data_Preprocessing.EncodingFeatures import Encoding
from Data_Preprocessing.ScalingFeatures import Scaling
from Data_Preprocessing.BalancingData import Balancing
from Data_Models.RFC_Model import Train_Pred_Submit_Model


app = Flask(__name__)

# Home Page ------------------------------------------------------------------------------------------------
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
@app.route("/", methods=['GET'])
def home():
    return render_template('home.html')

@app.route("/home_type", methods=['POST'])
def home_type():
    input_type = request.form['input_type']
    if input_type == 'Training_Model':
        # url_for('thyroidTraining')
        thyroidTraining()
    elif input_type == 'Predicting_Record':
        return render_template('home_Thyroid_Record.html')
    elif input_type == 'Predicting_File':
        return render_template("home_Thyroid_File.html")
    else:
        return render_template("home.html")

'''
@app.route("/home_Thyroid_Record", methods=['POST'])
def home_Thyroid_Record():
    return render_template('home_Thyroid_Record.html')

@app.route("/home_Thyroid_File", methods=['POST'])
def home_Thyroid_File():
    return render_template('home_Thyroid_File.html')
'''


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Thyroid Record  ---------------------------------------------------------------------------------------
@app.route("/predict_Thyroid_Record", methods=['POST'])
def predict_Thyroid_Record():
    results = {}
    th_features = ['age', 'TSH', 'T4U', 'FTI', 'T3', 'TT4',
                   'sex', 'on_thyroxine', 'query_hypothyroid', 'query_hyperthyroid', 'thyroid_surgery', 'tumor']
    in_features = [[request.form[col] for col in th_features]]
    X = pd.DataFrame(np.array(in_features), columns=th_features)  # Creating Data Frame with input values
    data = Scaling().MinMaxScaling(X, "predicting")  # Features Scaling using MinMaxScaler
    Y_pred = Train_Pred_Submit_Model().modelPredictions(data, "predicting")  # Predicting result

    if Y_pred[0] == 1:
        results["result"] = "Patient has THYROID"
    else:
        results["result"] = " Patient does not have THYROID"

    return render_template('results_Thyroid_Record.html', data = X, results = results)

# Thyroid File  ---------------------------------------------------------------------------------------
@app.route("/predict_Thyroid_File", methods=['POST'])
def predict_Thyroid_File():
    GVP = GlobalVariablesPath()

    results = {}
    in_file = request.files['in_file']  # Loading the file
    file_name = in_file.filename.split(".")[0] + "_Submission.csv"  # Extracting file name
    data = pd.read_csv(FileStorage(in_file).stream)

    Y_pred = thyroidPredicting(data)  # Predicting result
    results["result"] = "Prediction is completed successfully"

    Train_Pred_Submit_Model().modelSubmissions(Y_pred, file_name)  # Submitting the result into csv
    results["submission_file_path"] = GVP.filesPath["predictingFiles"] + file_name

    return render_template('results_Thyroid_File.html', results=results)

# ThyroidTraining  ---------------------------------------------------------------------------------------
@app.route("/thyroidTraining", methods=['POST'])
def thyroidTraining():
    GVP = GlobalVariablesPath()

    data = pd.read_csv(GVP.filesPath["trainingFile"])   # Loading Data
    data = FeatureSelection().selectedFeatures(data)    # Feature Selection
    data = Cleaning().cleanData(data)                   # Data Cleaning
    data = Encoding().convert_Cat_to_Num(data)          # Features Encoding
    data = Scaling().MinMaxScaling(data, "training")    # Features Scaling using MinMaxScaler
    data = Balancing().overSampling(data)  # Oversampling the data
    Train_Pred_Submit_Model().modelTraining(data)       # Training the Model

    results = {}
    results["result"] = "Model is trained Successfully"

    return render_template('results_Page.html', results=results)


# ThyroidPredicting  ---------------------------------------------------------------------------------------
@app.route("/thyroidPredicting", methods=['POST'])
def thyroidPredicting(data):
    GVP = GlobalVariablesPath()

    data = FeatureSelection().selectedFeatures(data)        # Feature Selection
    data = Cleaning().cleanData(data)                     # Data Cleaning
    data = Encoding().convert_Cat_to_Num(data)            # Features Encoding
    data = Scaling().MinMaxScaling(data, "predicting")      # Features Scaling using MinMaxScaler
    Y_pred = Train_Pred_Submit_Model().modelPredictions(data, "predicting")  # Predicting result
    return Y_pred

if __name__ == "__main__":
    app.run(debug=True, port=7654)