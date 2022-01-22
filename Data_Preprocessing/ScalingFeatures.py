
# ------------------------------- System defined Packages -------------------------------
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import joblib

# ------------------------------- User defined Packages -------------------------------
from GlobalVariables.GlobalVariables import GlobalVariablesPath

class Scaling:
    def __init__(self):
        self.GVP = GlobalVariablesPath()

    # Method 1 ------------------------------------------------------------------------------------------
    def MinMaxScaling(self, data, validate):
        if "hypothyroid" in data.columns:
            X = data.drop("hypothyroid", axis=1)
            Y = data["hypothyroid"]
        else:
            X = data.copy()

        if validate == "training":
            MMS = MinMaxScaler()
            MMS.fit(X)
            joblib.dump(MMS, self.GVP.filesPath["PickleFiles"] + "MinMaxScalar_thyroid.pkl")
        else:
            MMS = joblib.load(self.GVP.filesPath["PickleFiles"] + "MinMaxScalar_thyroid.pkl")

        X_array = MMS.transform(X)
        X_scale = pd.DataFrame(X_array, columns=X.columns)

        if "hypothyroid" in data.columns:
            data = pd.concat([X_scale, Y], axis=1)
        else:
            data = X_scale

        return data

