
# ------------------------------- System defined Packages -------------------------------
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import joblib
import pandas as pd

# ------------------------------- User defined Packages -------------------------------
from GlobalVariables.GlobalVariables import GlobalVariablesPath


class Train_Pred_Submit_Model:
    def __init__(self):
        self.GVP = GlobalVariablesPath()

    def thyroidModelReport(self, Y, Y_pred_prob, Y_pred, validate):
        # Classification Reports using Confusion Matrix ----------------------------------------------------------------
        roc_auc = roc_auc_score(Y, Y_pred_prob[:, 1])

        cm = confusion_matrix(Y, Y_pred)
        TP = cm[0, 0]
        TN = cm[1, 1]
        FP = cm[0, 1]
        FN = cm[1, 0]
        accuracy = (TP + TN) / float(TP + TN + FP + FN)
        error = (FP + FN) / float(TP + TN + FP + FN)
        precision = TP / float(TP + FP)
        recall = TP / float(TP + FN)
        specificity = TN / (TN + FP)
        TPR = TP / float(TP + FN)
        FPR = FP / float(FP + TN)

        # Loading 'XGBC_Model_Report.csv' file -------------------------------------------------------------------------
        RFC_Models_Report = pd.read_csv(self.GVP.filesPath["dataSet"] + "RFC_Models_Report.csv")

        # Storing classification reports in a csv file -----------------------------------------------------------------
        RFC_Models_Report.loc[validate, :] = [validate, roc_auc, TP, TN, FP, FN, accuracy, error, precision, recall,
                                              specificity, TPR, FPR]
        RFC_Models_Report.to_csv(self.GVP.filesPath["dataSet"] + "RFC_Models_Report.csv", index=False)


    def modelPredictions(self, data, validate):
        if "hypothyroid" in data.columns:
            X = data.drop('hypothyroid', axis=1)
            Y = data['hypothyroid']
        else:
            X = data.copy()

        # Predictions ==============================================================
        model = joblib.load(self.GVP.filesPath["PickleFiles"] + "RandomForestClassifier.pkl")
        Y_pred_prob = model.predict_proba(X)
        Y_pred = model.predict(X)

        # Model Report Summary ----------------------------------------------------------
        if "hypothyroid" in data.columns:
            self.thyroidModelReport(Y, Y_pred_prob, Y_pred, validate)

        return Y_pred

    def modelTraining(self, data):
        X = data.drop('hypothyroid', axis=1)
        Y = data['hypothyroid']
        model = RandomForestClassifier()

        # Training Model ===========================================================
        model.fit(X, Y)

        # Storing Model as a pickle ================================================
        joblib.dump(model, self.GVP.filesPath["PickleFiles"] + "RandomForestClassifier.pkl")

        # Creating a Data Frame to store Classification reports ====================
        # This dataset is used to store the performance of the models
        RFC_Models_Report = pd.DataFrame(columns=["data", "roc_auc", "TP", "TN", "FP", "FN", "accuracy", "error", "precision", "recall", "specificity", "TPR", "FPR"])
        RFC_Models_Report.to_csv(self.GVP.filesPath["dataSet"] + "RFC_Models_Report.csv", index=False)

        # Getting predictions and storing in a DataFrame ===========================
        Y_pred_prob = model.predict_proba(X)
        Y_pred = self.modelPredictions(data, "training")
        self.thyroidModelReport(Y, Y_pred_prob, Y_pred, "training")

    def modelSubmissions(self, Y_pred, file_name):

        submission = pd.read_csv(self.GVP.filesPath["Temp"] + "submission.csv")
        submission["TY_Prediction"] = Y_pred
        submission.to_csv(self.GVP.filesPath["predictingFiles"] + file_name, index=False)
