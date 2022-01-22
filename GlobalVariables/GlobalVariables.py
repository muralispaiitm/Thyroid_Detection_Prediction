

class GlobalVariablesPath:
    def __init__(self):
        self.filesPath = {"dataSet"         : "Files_Storage/DataSet/",
                          "trainingFile"    : "Files_Storage/DataSet/Training/training_Hypothyroid.csv",
                          "testingFile"     : "Files_Storage/DataSet/Testing/testing_Hypothyroid.csv",
                          "predictingFiles" : "Files_Storage/DataSet/Predicting/",
                          "PickleFiles"     : "Files_Storage/PickleFiles/",
                          "Figures"         : "Files_Storage/Figures",
                          "Temp"            : "Files_Storage/Temp/"
                          }
        self.syntaxErrorVal = "?"
        self.CategoricalFeatures = ['sex', 'on_thyroxine', 'query_hypothyroid', 'query_hyperthyroid', 'thyroid_surgery', 'tumor', 'hypothyroid']
        self.NumericalFeatures = ['age', 'TSH', 'T4U', 'FTI', 'T3', 'TT4']
        self.SelectedFeatures = ['age', 'TSH', 'T4U', 'FTI', 'T3', 'TT4', 'sex', 'on_thyroxine', 'query_hypothyroid', 'query_hyperthyroid', 'thyroid_surgery', 'tumor', 'hypothyroid']

        self.MySQL_Variables = {"host"          : 'localhost',
                                "user"          : "root",
                                "pwd"           : "password",
                                "database"      : "minbankproduct",
                                "trainingTable" : "training_BOP",
                                "testingTable"  : "testing_BOP"
                                }
