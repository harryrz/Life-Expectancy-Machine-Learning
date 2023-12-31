import pandas as pd #importing pandas data framework
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # for standardizing numerial features
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

#Data Loading and Observing----------------------------------
dataset = pd.read_csv('life_expectancy.csv'); #read in csv file and store it in a dataframe called dataset
print(dataset.head()); #checking first five columns
print(dataset.describe()); #printing the summary of dataframe

dataset = dataset.drop(['Country'], axis = 'columns'); #dropping the country column from the dataset

labels = dataset.iloc[:, -1]; # selecting all the rows(:), and access the last column (-1);
features = dataset.iloc[:, 0:-1]; # selecting all the rows(:), and access columns from 0 to the last column not including it

#Data Preprocessing-------------------------------------------
features = pd.get_dummies(features); # Applying one-hot state encoding for categorical values in the DataFrame

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.25, random_state=42); #Spliting data into training and testing sets using sci-kit learn

numerical_features = features.select_dtypes(include=['float64', 'int64']); # This makes sure all the numerical features in the features framework gets captured, both floating numbers and integer numbers, numerical_features is a framework
numerical_columns = numerical_features.columns

ct = ColumnTransformer([("only numeric", StandardScaler(), numerical_columns)], remainder='passthrough')

features_train_scaled = ct.fit_transform(features_train)
features_test_scaled = ct.transform(features_test); # we cannot use fit_transform on the testing dataset because it may introduce potential biases

#Model Construction-------------------------------------------
my_model = Sequential()
input = InputLayer(input_shape = (features.shape[1], ))
my_model.add(input); # constructing input layer
my_model.add(Dense(32, activation = 'relu')); # constructing hidden layer
my_model.add(Dense(1)); # constructing output layer

print(my_model.summary())

#Optimizer and Model Compilation------------------------------
opt = Adam(learning_rate = 0.01)
my_model.compile(loss='mse', metrics=['mae'], optimizer = opt)

#Model Fitting and Evalutation--------------------------------
my_model.fit(features_train_scaled, labels_train, epochs = 40, batch_size = 1, verbose = 1)
res_mse, res_mae = my_model.evaluate(features_test_scaled, labels_test, verbose = 0)
print(res_mse, res_mae)






