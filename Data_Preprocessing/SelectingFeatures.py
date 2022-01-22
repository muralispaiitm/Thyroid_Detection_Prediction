

# ------------------------------- User defined Packages -------------------------------
from GlobalVariables.GlobalVariables import GlobalVariablesPath


class FeatureSelection:
  def __init__(self):
    self.GVP = GlobalVariablesPath()

  def selectedFeatures(self, data):
    selectedFeatures = self.GVP.SelectedFeatures

    if "hypothyroid" not in data.columns:
      selectedFeatures.remove("hypothyroid")

    data = data[selectedFeatures]
    return data