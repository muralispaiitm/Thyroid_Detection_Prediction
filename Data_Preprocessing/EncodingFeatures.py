
# ------------------------------- System defined Packages -------------------------------
import pandas as pd


# ------------------------------- User defined Packages -------------------------------
from GlobalVariables.GlobalVariables import GlobalVariablesPath


class Encoding:
    def __init__(self):
        self.GVP = GlobalVariablesPath()

    def convert_Cat_to_Num(self, data):

        # 'sex' column
        data["sex"] = data["sex"].map({"F": 0, "M": 1})

        # 'hypothyroid' column
        if "hypothyroid" in data.columns:
            data["hypothyroid"] = [0 if ((val == "N") | (val == "negative")) else 1 for val in data["hypothyroid"]]

        # Remaining other columns
        convertFeatures = self.GVP.CategoricalFeatures
        convertFeatures.remove('sex')
        convertFeatures.remove("hypothyroid")
        for feature in convertFeatures:
            data[feature] = data[feature].map({"f": 0, "t": 1})

        # Converting numbers from 'object' into numerics
        data = data.apply(pd.to_numeric)

        return data