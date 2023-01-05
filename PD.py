#Importing the libraries for data manipulation and visualization.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import datetime
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import RandomOverSampler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import pickle
warnings.filterwarnings('ignore')

#Importing the dataset.
dataset = pd.read_csv('./logreg_data.csv', delimiter = ';')

#1000 observations, 7 features and 1 target.
dataset

#Missing values for loan amount and city features.
#Combination of categorical (both nominal and ordinal) and continuous features.
#Target is dichotomous.
dataset.info()

#Calculating the age of particular loan applicant based on his/her year of birth and actual year.
act_year = datetime.date.today().year
dataset['Age'] = act_year - dataset['Year of birth']


######
#PLOTS
######


#Distribution of default.
plt.figure(figsize = (10, 10))
sns.countplot(dataset['DEFAULT'], palette = 'Set1')
plt.title("Distribution of default")
plt.savefig('Distribution_of_default.png')
plt.show()

#Histogram of age - with and without negative ages.
fig, ax = plt.subplots(1, 2, figsize=(25, 10))
sns.distplot(dataset['Age'], color = "purple", ax = ax[0])
ax[0].set_title("Distribution of loan applicants' ages")
sns.distplot(dataset['Age'][dataset['Age'] > 0], color = "green", ax =ax[1])
ax[1].set_title("Distribution of loan applicants' ages - corrected")
plt.savefig('Age_distribution.png')
plt.show()

#Age x Default
plt.figure(figsize = (15, 10))
sns.distplot(dataset[dataset['DEFAULT'] == 0][dataset['Age'] > 0]['Age'], color = "dodgerblue", label = "Non-default", kde = False)
sns.distplot(dataset[dataset['DEFAULT'] == 1][dataset['Age'] > 0]['Age'], color = "orange", label = "Default", kde = False)
plt.title("Distribution of loan applicants' ages based their default statuses")
plt.savefig('Age_Default_distribution.png')
plt.legend()
plt.show()

#Pairplot for continuous features with respect to default.
plt.figure(figsize = (20, 20))
sns.pairplot(dataset.drop(['Year of application','Year of birth'], axis = 1)[dataset['Age'] > 0], hue = 'DEFAULT', palette = 'coolwarm')
plt.savefig('Pairplot.png')
plt.show()


###################
#DATA PREPARATION
###################


#Replacing the outliers within Age feature with missing value (N/A).
dataset.loc[(dataset['Age'] < 0), 'Age'] = np.nan

#Calculating the years since the application based on the year of application.
dataset['Years since application'] = act_year - dataset['Year of application']


#Dropping the redundant features Year of birth and Year of application from the dataset.
dataset.drop(['Year of birth','Year of application'], axis = 1, inplace = True)

#Reordering the dataset.
dataset = dataset[['Loan term', 'Loan Amount', 'Years since application',
                   'Age', 'City', 'Gender', 'Education', 'DEFAULT']]

#Extracting the target variable and the features.
Target = dataset[['DEFAULT']]
Features = dataset.drop('DEFAULT', axis = 1)

#Splitting the data into training set, validation set and test set.
seed = 1111
X_temp, X_test, y_temp, y_test = train_test_split(Features, Target,
                                                  test_size = 0.2, stratify = Target, random_state = seed)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp,
                                                  test_size = 0.125, stratify = y_temp, random_state = seed)

#Due to the imbalanced class of the target variable, the oversampling of the training set will be performed.
ros = RandomOverSampler(random_state = seed)
X_train_os, y_train_os = ros.fit_resample(X_train, y_train)

#As we can see, the class is already balanced.
y_train_os.value_counts()

#Doublechecking the overview of missing values within this dataset.
X_train_os.info()

#Imputing missing values with a mode by using the Simple Imputer:
imp_mode = SimpleImputer(strategy = 'most_frequent')

#Extracting the categorical features' names.
X_categorical = X_train_os.describe(include = 'object').columns.values

for cat in X_categorical:
    X_train_os[cat] = imp_mode.fit_transform(X_train_os[[cat]])
    for lst in[X_val, X_test]:
        lst[cat] = imp_mode.transform(lst[[cat]])

#Ordinal encoding of categorical ordinal feature(s).
oe = OrdinalEncoder()

X_train_os['Education'] = oe.fit_transform(X_train_os[['Education']]).astype('int')
for lst in [X_val, X_test]:
    lst['Education'] = oe.transform(lst[['Education']]).astype('int')

#Creating my own function fo dummy encoding based on the pandas.get_dummies function.
def dummy_encoding(dataset, var_list):
    new_dataset = dataset.copy()
    for var in var_list:
        #Get dummy variables for each category within the passed feature from the passed dataset which will have defined prefix.
        var_dummy=pd.get_dummies(dataset[var], prefix = str(var))
        #(G = number of categories wihtin variable) --> hence we are extracting (G-1) dummies as the one excluded dummy will be the baseline, in order to deal with the multicolinearity.
        var_dummy = var_dummy.iloc[:, 1:]
        #Concatenating the passed dataset with the dummies and dropping the initial feature which has been encoded.
        new_dataset = pd.concat((new_dataset, var_dummy), axis = 1).drop(var, axis = 1)                             
    return new_dataset

#Dummy encoding
X_train_dummy = dummy_encoding(X_train_os, ['City', 'Gender'])
X_val_dummy = dummy_encoding(X_val, ['City', 'Gender'])
X_test_dummy = dummy_encoding(X_test, ['City', 'Gender'])

###
#DATA PREPROCESSING OF CONTINUOUS FEATURES
###

X_continuous = ['Loan term', 'Loan Amount', 'Years since application', 'Age']

#Imputing the missing values with a mean by using the Simple ImputER:
imp_mean = SimpleImputer()

for cont in X_continuous:
    X_train_dummy[cont] = imp_mode.fit_transform(X_train_dummy[[cont]])
    for lst in[X_val_dummy, X_test_dummy]:
        lst[cont] = imp_mode.transform(lst[[cont]])

#Discretization of continuous features into 5 bins
bindis = KBinsDiscretizer(n_bins = 5, encode = 'onehot-dense')

X_train_final = X_train_dummy.copy().reset_index(drop = True)
X_val_final = X_val_dummy.copy().reset_index(drop = True)
X_test_final = X_test_dummy.copy().reset_index(drop = True)

for cont in X_continuous:

    X_train_arr = bindis.fit_transform(X_train_dummy[[cont]])

    X_val_arr = bindis.transform(X_val_dummy[[cont]])
    X_test_arr = bindis.transform(X_test_dummy[[cont]])

    bins_ints = [f'{cont}_{bindis.bin_edges_[0][i]}_{bindis.bin_edges_[0][i+1]}' for i in range(len(bindis.bin_edges_[0])-1)]

    X_train_bins = pd.DataFrame(X_train_arr, columns = bins_ints).astype('int').reset_index(drop = True)
    X_train_final = pd.concat((X_train_final, X_train_bins), axis = 1).drop(cont, axis = 1, errors = 'ignore')

    X_val_bins = pd.DataFrame(X_val_arr, columns = bins_ints).astype('int').reset_index(drop = True)
    X_val_final = pd.concat((X_val_final, X_val_bins), axis = 1).drop(cont, axis = 1, errors = 'ignore')

    X_test_bins = pd.DataFrame(X_test_arr, columns = bins_ints).astype('int').reset_index(drop = True)
    X_test_final = pd.concat((X_test_final, X_test_bins), axis = 1).drop(cont, axis = 1, errors = 'ignore')

#exporting preprocessed training, validation and test sets.
for feats, targets, name in zip([X_train_final, X_val_final, X_test_final], [y_train_os, y_val, y_test], ['train', 'val', 'test']):
    exp_df = pd.concat((feats.reset_index(drop = True), targets.reset_index(drop = True)), axis = 1)
    exp_df.to_csv(f'{name}_set_preprocessed.csv', index = False)


###################
#MODELLING
###################

#Logistic regression model initialization
lr = LogisticRegression(random_state = seed)

#Hyperparameter space deifinition of logistic regression
params_grid = {
               'penalty': ['l2', 'l1', 'elasticnet', None],
                'class_weight': ['balanced', None],
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                'C': np.linspace(0.001, 1.0, 1000),
                'warm_start': [True, False]
               }

#Creating the grid search cross validation object with 10 folds.
grcv = GridSearchCV(estimator = lr, param_grid = params_grid,
                    scoring = 'roc_auc', n_jobs = -1,cv = 10, verbose = 1)

#Fitting the grid search using the training set
grcv.fit(X_train_final, y_train_os)

#Obtaining the model with the best hyperparameters' values for logistic regression.
final_model = grcv.best_estimator_
final_model

#Fitting the new logistic regression model on joined training and validation data.
y_train_valid = pd.concat((y_train_os.reset_index(drop = True), y_val.reset_index(drop = True)))
X_train_valid = pd.concat((X_train_final, X_val_final))
final_model.fit(X_train_valid, y_train_valid)

#Saving the model
pickle.dump(final_model, open('LR_final.pkl', 'wb'))
    



############
#EVALUATION
############


#Storing printing the auc scores for training and validation set.
auc_train = roc_auc_score(y_train_os, final_model.predict_proba(X_train_final)[:, 1])
auc_val = roc_auc_score(y_val, final_model.predict_proba(X_val_final)[:, 1])
auc_train_val = roc_auc_score(y_train_valid, final_model.predict_proba(X_train_valid)[:, 1])
auc_df = pd.DataFrame(zip(['training', 'validation', 'training_validation'], [auc_train, auc_val, auc_train_val]), columns = ['set', 'auc'])
auc_df

#AUC score on test set.
auc_test = roc_auc_score(y_test, final_model.predict_proba(X_test_final)[:, 1])

auc_df = pd.concat((auc_df, pd.DataFrame(zip(['test'], [auc_test]), columns = ['set', 'auc'])))
auc_df

#Saving the AUC results
auc_df.to_csv('auc_results.csv', index = False)


#Printing the confusion matrix and the classifion report containing precision and recall.
conf_mtrx = pd.DataFrame(confusion_matrix(y_test, final_model.predict(X_test_final))).rename(
                                            columns = {0: 'Predicted - Non-default',1: 'Predicted - Default'},
                                            index = {0: 'Actual - Non-default', 1: 'Actual - Default'})
conf_mtrx

#saving the confusion matrix result
conf_mtrx.to_csv('confusion_matrix.csv')

#Plotting the ROC curve for test set.
plt.figure(figsize = (10,10))
fpr_train, tpr_train, _ = roc_curve(y_test, final_model.predict_proba(X_test_final)[:,1])
fpr_val, tpr_val, _ = roc_curve(y_test, final_model.predict_proba(X_test_final)[:,1])
fpr_train_val, tpr_train_val, _ = roc_curve(y_test, final_model.predict_proba(X_test_final)[:,1])
fpr_test, tpr_test, _ = roc_curve(y_test, final_model.predict_proba(X_test_final)[:,1])
plt.plot(fpr_train, tpr_train, label = 'training', color = 'brown')
plt.plot(fpr_val, tpr_val, label = 'validation', color = 'purple')
plt.plot(fpr_train_val, tpr_train_val, label= 'training & validation', color = 'blue')
plt.plot(fpr_test, tpr_test, label = 'test', color = 'green')
plt.plot([0, 1], [0, 1], 'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC curves')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('ROC_curve.png')
plt.show()

#Storing the intercept and regressor coefficients' values into DataFrame
intercept = pd.DataFrame(zip(['intercept'], final_model.intercept_), columns = ['variable', 'coefficient'])
coefficients = pd.DataFrame(final_model.coef_.reshape(len(X_train_valid.columns), 1), columns = ['coefficient'])
feat_cols = pd.DataFrame(X_train_valid.columns, columns = ['variable'])
feat_coefs = pd.concat((intercept, pd.concat((feat_cols, coefficients),axis = 1)))
feat_coefs

#Saving the coefficients magnitudes
feat_coefs.to_csv('beta_coefficients.csv', index = False)

#Plotting the beta coefficients
fig, ax = plt.subplots(figsize = (15, 10))
sns.barplot(data = feat_coefs, y = 'variable', x = 'coefficient', ax = ax)
ax.tick_params(axis = 'x', which = 'major', labelsize = 9, rotation = 90)
plt.title('Logistic regression - beta coefficients')
plt.xlabel('Beta coefficients')
plt.ylabel('Variable names')
plt.savefig('Beta_coefficients.png')
plt.show()
# %%
