import pandas as pd
from scipy import stats
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.linear_model import  Ridge
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures , LabelEncoder
from sklearn.model_selection import cross_val_score
%matplotlib inline


flight_df = pd.read_csv("/Users/randalllionelkharkrang/Desktop/Innopolis/Semester1/Machine_learning/Assignment_1/flight_delay.csv",parse_dates=['Scheduled depature time','Scheduled arrival time'])

#flight_df.head() # just to check the data
#print(flight_df.shape) to get dimension of data matrix
#flight_df.describe() # to get summary statistics

# the following code extracts the following:
 # Scheduled departure month, flight duraiton
 #scheduled day of week, scheduled departure hour
 # scheuled departure year
flight_df['Flight duration(minutes)'] = (abs(flight_df['Scheduled depature time'] - flight_df['Scheduled arrival time'])).dt.seconds/60
flight_df['Scheduled depature month'] = flight_df['Scheduled depature time'].dt.month
flight_df['Scheduled depature dow'] = flight_df['Scheduled depature time'].dt.day_name()
flight_df['Scheduled depature hour'] = flight_df['Scheduled depature time'].dt.hour
flight_df['Scheduled depature year'] = flight_df['Scheduled depature time'].dt.year

#plots Flight duration vs delay
flight_df.plot(x='Flight duration(minutes)', y='Delay', style='o')
plt.title('Duration vs Delay')
plt.xlabel('Duration')
plt.ylabel('Delay')
plt.show()

# splitting the data based on scheduled departure year
X_train = flight_df.loc[(flight_df['Scheduled depature year'] >= 2015) & (flight_df['Scheduled depature year'] <=2017)]
X_test = flight_df.loc[flight_df['Scheduled depature year'] == 2018]
y_train = X_train['Delay']
y_test = X_test['Delay']
X_train = X_train['Flight duration(minutes)']
X_test = X_test['Flight duration(minutes)']


# the following handles outlier detection for simple linear regression
df = pd.concat([X_train,y_train],axis=1)
df_mean, std_dev = np.mean(df['Flight duration(minutes)']), np.std(df['Flight duration(minutes)'])
cutoff = std_dev*3
lower,upper = df_mean - cutoff, df_mean + cutoff
# identify outliers
outliers = [x for x in df['Flight duration(minutes)'] if x < lower or x > upper]
print('Identified outliers: %d' % len(outliers))
# remove outliers
df.drop(df[df['Flight duration(minutes)'] < lower].index, inplace = True)
df.drop(df[df['Flight duration(minutes)'] > upper].index, inplace = True)
print('Non-outlier observations: %d' % len(df))


#seperate the data set again
X_train = df['Flight duration(minutes)']
y_train = df['Delay']
print(X_train.shape)
print(y_train.shape)

#reshaping te data
X_train = (X_train.to_numpy()).reshape(-1,1)
X_test = (X_test.to_numpy()).reshape(-1,1)

# getting the linear regression model class
linreg = LinearRegression()
linreg.fit(X_train,y_train)
print(f"model intercept: {linreg.intercept_}")
print(f"model slope: {linreg.coef_}")

#printing the training error for various metrics
y_train_pred  = linreg.predict(X_train)
print('Mean Absolute Error training :', metrics.mean_absolute_error(y_train_pred, y_train))
print('Mean Squared Error training :', metrics.mean_squared_error(y_train, y_train_pred))
print('Root Mean Squared Error training :', np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)))
print('R2 score training :', metrics.r2_score(y_train,y_train_pred))

#printing the test error for various metrics
y_train_pred  = linreg.predict(X_train)
print('Mean Absolute Error training :', metrics.mean_absolute_error(y_train_pred, y_train))
print('Mean Squared Error training :', metrics.mean_squared_error(y_train, y_train_pred))
print('Root Mean Squared Error training :', np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)))
print('R2 score training :', metrics.r2_score(y_train,y_train_pred))



##--------Polynomial regression ------ ####
#Extract data as it was converted to numpy before
X_train = flight_df.loc[(flight_df['Scheduled depature year'] >= 2015) & (flight_df['Scheduled depature year'] <=2017)]
X_test = flight_df.loc[flight_df['Scheduled depature year'] == 2018]
y_train = X_train['Delay']
y_test = X_test['Delay']
X_train = X_train['Flight duration(minutes)']
X_test = X_test['Flight duration(minutes)']

#same outlier detection method as simple linear regression
df = pd.concat([X_train,y_train],axis=1)
df_mean, std_dev = np.mean(df['Flight duration(minutes)']), np.std(df['Flight duration(minutes)'])
cutoff = std_dev*3
lower,upper = df_mean - cutoff, df_mean + cutoff
# identify outliers
outliers = [x for x in df['Flight duration(minutes)'] if x < lower or x > upper]
print('Identified outliers: %d' % len(outliers))
# remove outliers
df.drop(df[df['Flight duration(minutes)'] < lower].index, inplace = True)
df.drop(df[df['Flight duration(minutes)'] > upper].index, inplace = True)
print('Non-outlier observations: %d' % len(df))

#seperate the data set again
X_train = df['Flight duration(minutes)']
y_train = df['Delay']

#reshaping the data
X_train = (X_train.to_numpy()).reshape(-1,1)
X_test = (X_test.to_numpy()).reshape(-1,1)

#use CV to find degree that gives smallest error
#use pipeline to ease code
#used negative MSE, but converted it later
degrees = [1, 5, 9]
plt.figure(figsize=(14, 5))
for i in range(len(degrees)):
    ax = plt.subplot(1, len(degrees), i + 1)
    plt.setp(ax, xticks=(), yticks=())

    polynomial_features = PolynomialFeatures(degree=degrees[i])
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    pipeline.fit(X_train, y_train)

    # Evaluate the models using crossvalidation
    scores = cross_val_score(pipeline, X_train, y_train,
                             scoring="neg_mean_squared_error", cv=5)

    plt.scatter(X_test, pipeline.predict(X_test), label="Model")
    plt.scatter(X_test, y_test, edgecolor='b',label="Actual function",alpha=0.3)
    # plt.scatter(X, y, edgecolor='b', s=20, label="Samples")
    plt.xlabel("x")
    plt.ylabel("y")
    # plt.xlim((0, 1))
    # plt.ylim((-2, 2))
    plt.legend(loc="best")
    plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
        degrees[i], -scores.mean(), scores.std()))
plt.show()





##-------Multiplie Linear Regresssion with Ridge -----###

#Splitting the data and using multiple columns as features
X_test = flight_df.loc[flight_df['Scheduled depature year'] == 2018]
X_train = flight_df.loc[(flight_df['Scheduled depature year'] >= 2015) & (flight_df['Scheduled depature year'] <=2017)]
y_train = X_train['Delay']
y_test = X_test['Delay']

X_train = X_train.drop(['Scheduled depature time','Scheduled arrival time'],axis=1)
X_test = X_test.drop(['Scheduled depature time','Scheduled arrival time'],axis=1)

#split into validation set to find optimal alpha value
#when using regularization
x_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1/8, random_state=123)

print("Number of missing values in train before encoding: ",x_train.isnull().sum().sum())
print("Number of missing values in test before encoding: ",X_test.isnull().sum().sum())
print("Number of missing values in train before encoding: ",x_val.isnull().sum().sum())

#use label encoder and stack into single column
encoder = LabelEncoder()
feats = ['Depature Airport','Destination Airport','Scheduled depature dow']
encoder.fit(x_train[feats].stack().unique())

#transform training data
x_train['Depature Airport']=encoder.transform(x_train['Depature Airport'])
x_train['Destination Airport']= encoder.transform(x_train['Destination Airport'])
x_train['Scheduled depature dow']= encoder.transform(x_train['Scheduled depature dow'])

#extract dictionary consisting of transformed classses
encoder_dict = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

#transform the validation and test set
x_val['Depature Airport']=x_val['Depature Airport'].apply(lambda x: encoder_dict.get(x,-1))
x_val['Destination Airport']= x_val['Destination Airport'].apply(lambda x: encoder_dict.get(x,-1))
x_val['Scheduled depature dow']= x_val['Scheduled depature dow'].apply(lambda x: encoder_dict.get(x,-1))
X_test['Depature Airport']=X_test['Depature Airport'].apply(lambda x: encoder_dict.get(x,-1))
X_test['Destination Airport']= X_test['Destination Airport'].apply(lambda x: encoder_dict.get(x,-1))
X_test['Scheduled depature dow']= X_test['Scheduled depature dow'].apply(lambda x: encoder_dict.get(x,-1))

print("Number of missing values in train after encoding: ",x_train.isnull().sum().sum())
print("Number of missing values in validation after encoding: ",x_val.isnull().sum().sum())
print("Number of missing values in Test after encoding: ",X_test.isnull().sum().sum())

#impute the dataset
imputer = SimpleImputer(strategy='mean')
imputer.fit(x_train)
x_train = imputer.transform(x_train)
X_test = imputer.transform(X_test)
x_val = imputer.transform(x_val)


print("Number of missing values in train after imputing: ",np.sum(np.sum(np.isnan(x_train))))
print("Number of missing values in val after imputing: ",np.sum(np.sum(np.isnan(x_val))))

print("Number of missing values in test after imputing: ",np.sum(np.sum(np.isnan(X_test))))

#scaling the feaures
scaler = preprocessing.MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
X_test = scaler.transform(X_test)
x_val = scaler.transform(x_val)

#Detect and remove outliers
from sklearn.neighbors import LocalOutlierFactor
# identify outliers in the training dataset
lof = LocalOutlierFactor()
yhat = lof.fit_predict(x_train)
# select all rows that are not outliers
mask = yhat != -1
x_train, y_train = x_train[mask, :], y_train[mask]
# summarize the shape of the updated training dataset
print(x_train.shape, y_train.shape)


#Use PCA to get better idea of data
pca = PCA(n_components=8)
#pca.fit(X_train)
x_reduced = pca.fit_transform(x_train)

print(pca.explained_variance_ratio_)


#use the ridge class
regressor = Ridge()
regressor.fit(x_train, y_train)
print(f"Model coefficients : {regressor.coef_}")

#find out best alpha value from validation set
alphas = [2.2, 2, 1.5, 1.3, 1.2, 1.1, 1, 0.3, 0.1]
losses = []
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(x_train,y_train)
    predict_val = ridge.predict(x_val)
    mse = metrics.mean_squared_error(y_val,predict_val)
    losses.append(mse)

plt.plot(alphas, losses)
plt.title("Ridge alpha value selection")
plt.xlabel("alpha")
plt.ylabel("Mean squared error")
plt.show()

best_alpha = alphas[np.argmin(losses)]
print("Best value of alpha:", best_alpha)
print("MSE : ", losses[np.argmin(losses)] )

#get the test error using various metrics
regressor = Ridge(best_alpha)
regressor.fit(x_train, y_train)
y_pred = regressor.predict(X_test)
print('Mean Absolute Error for ridge for test :', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error ridge for test :', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error for ridge for test:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2 score  for test :', metrics.r2_score(y_test,y_pred))


#Get the training error for various metrics
regressor = Ridge(best_alpha)
regressor.fit(x_train, y_train)
y_pred_train = regressor.predict(x_train)
print('Mean Absolute Error for ridge for train:', metrics.mean_absolute_error(y_train, y_pred_train))
print('Mean Squared Error ridge for train:', metrics.mean_squared_error(y_train, y_pred_train))
print('Root Mean Squared Error for ridge for train:', np.sqrt(metrics.mean_squared_error(y_train, y_pred_train)))
print('R2 score  for train :', metrics.r2_score(y_train,y_pred_train))
