
# ------------------------------- System defined Packages -------------------------------
from imblearn.over_sampling import SMOTE
import pandas as pd

# ------------------------------- User defined Packages -------------------------------

class Balancing:
    def __init__(self):
        pass

    def overSampling(self, data):
        X = data.drop("hypothyroid", axis=1)
        Y = data["hypothyroid"]

        Oversampler_Smote = SMOTE(random_state=12)
        X_overSample, Y_overSample = Oversampler_Smote.fit_resample(X, Y)
        data = pd.concat([X_overSample, Y_overSample], axis=1)
        return data