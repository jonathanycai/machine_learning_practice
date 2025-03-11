import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# import data
global_temp = pd.read_csv('data/GlobalTemperatures.csv')

# exploratory data analysis
# print(global_temp.shape)
# print(global_temp.columns)
# print(global_temp.info())
# print(global_temp.isnull().sum())

# function to convert celsius to fahrenheit
def converttemp(x):
    x = (x * 1.8) + 32
    return float(x)

# function to create copy of dataframe and drop columns with high cardinality
def wrangle(df):
    df = df.copy()
    df = df.drop(columns=["LandAverageTemperatureUncertainty", "LandMaxTemperatureUncertainty",
                          "LandMinTemperatureUncertainty", "LandAndOceanAverageTemperatureUncertainty"], axis=1)
    
    # convert temperatures to fahrenheit
    df["LandAverageTemperature"] = df["LandAverageTemperature"].apply(converttemp)
    df["LandMaxTemperature"] = df["LandMaxTemperature"].apply(converttemp)
    df["LandMinTemperature"] = df["LandMinTemperature"].apply(converttemp)
    df["LandAndOceanAverageTemperature"] = df["LandAndOceanAverageTemperature"].apply(converttemp)

    df["dt"] = pd.to_datetime(df["dt"])
    df["Month"] = df["dt"].dt.month
    df["Year"] = df["dt"].dt.year
    df = df.drop(columns=["dt"], axis=1)
    df = df.drop("Month", axis=1)
    df = df[df.Year >= 1850]
    df = df.set_index(["Year"])
    df = df.dropna()

    return df
    

global_temp = wrangle(global_temp)
print(global_temp.head())

# plot to find correlations in the data
corrMatrix = global_temp.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()

# separate data into feature/ targets
target = "LandAndOceanAverageTemperature"
y = global_temp[target]
x = global_temp[["LandAverageTemperature", "LandMaxTemperature", "LandMinTemperature"]]

# split data into training and validation sets
xtrain, xval, ytrain, yval = train_test_split(x, y, test_size = 0.2, random_state = 42)
print(xtrain.shape)
print(xval.shape)
print(ytrain.shape)
print(yval.shape)

# baseline model to beat
ypred = np.full(len(yval), ytrain.mean())  # Create baseline predictions for yval
print("Baseline MAE: ", round(mean_squared_error(yval, ypred), 5))

forest = make_pipeline(
    SelectKBest(k = "all"),
    StandardScaler(),
    RandomForestRegressor(
        n_estimators = 100,
        max_depth = 50,
        random_state=77,
        n_jobs=-1
    )
)

forest.fit(xtrain, ytrain)

errors = abs(ypred - yval)
mape = 100 * (errors / ytrain)
accuracy = 100 - np.mean(mape)
print("Random Forest Model: ", round(accuracy, 2), "%")