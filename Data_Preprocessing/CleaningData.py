
# ------------------------------- System defined Packages -------------------------------
import pandas as pd
import numpy as np

# ------------------------------- User defined Packages -------------------------------
from GlobalVariables.GlobalVariables import GlobalVariablesPath


class Cleaning:
    def __init__(self):
        self.GVP = GlobalVariablesPath()

    def cleanData(self, data):
        # Run removeRowsCols(),imputeMissingData()
        data = self.removeRowsCols(data)
        data = self.imputeSyntaxErrors(data)
        data = self.imputeMissingData(data)
        return data

    def removeRowsCols(self, data):
        # 1.1.1 Removing unnecessary Rows and Columns (NullRows, NUllColumns)

        # DROP ROWS
        nullIndices = data.index[(data.isnull().sum(axis=1)) / len(data.columns) > 0.7]
        data.drop(nullIndices, inplace=True)  # Droping the rows which have percentage of null values more than 70%.

        # DROP COLUMNS
        nullColumns = data.columns[data.isnull().sum() / len(data) > 0.7]
        data.drop(nullColumns, axis=1, inplace=True)  # Droping the columns which have percentage of null values more than 70%.

        data_Id = pd.DataFrame(data.index, columns = ["Id"])
        data_Id.to_csv(self.GVP.filesPath["Temp"] + "submission.csv", index=False)

        return data

    def imputeSyntaxErrors(self, data):
        # Imputing Syntax Errors
        syntaxVal = self.GVP.syntaxErrorVal  # Accessing syntax error value form the GlobalVariables class

        # Finding the number of syntax errors present in each column if it has and save the details
        syntaxErrors = data.isin([syntaxVal]).sum()
        syntaxErrors_cols = pd.DataFrame(syntaxErrors[syntaxErrors > 0]).T
        syntaxErrors_cols.to_csv(self.GVP.filesPath["dataSet"] + "thyroid_SyntaxErrors_Cols.csv")

        for col in syntaxErrors_cols.columns:
            syntaxIndex = data[data[col] == syntaxVal].index
            data.loc[syntaxIndex, col] = np.nan

        return data

    def imputeMissingData(self, data):
        # Number of NULLs present in each column and save the details
        nullValues = data.isin([np.nan]).sum()
        nullValues_cols = pd.DataFrame(nullValues[nullValues > 0]).T
        nullValues_cols.to_csv(self.GVP.filesPath["dataSet"] + "thyroid_NullValues_Cols.csv")

        # Replacing Null values
        data = data.fillna(method='bfill')
        data = data.fillna(method='ffill')  # Filling, if any NaN presents after "bfill"
        return data
