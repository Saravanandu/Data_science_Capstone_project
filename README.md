# Data_science_Capstone_project
## Health Care
import numpy as np import pandas as pd import seaborn as sns
import matplotlib.pyplot as plt 
df = pd.read_csv('F:/Data science/MC pjt/health care diabetes.csv') df
df.info()
df.isnull().sum().sum() 
df.duplicated().sum() 
df.describe().transpose() 
df.groupby(df['Glucose'] == 0).count() 
sns.histplot(df['Glucose']) 
df.groupby(df['BloodPressure'] == 0).count() 
sns.histplot(data=df['BloodPressure']) 
df.groupby(df['SkinThickness'] == 0).count() 
sns.histplot(data=df['SkinThickness'])
sns.histplot(data=df['Insulin']) 
df.groupby(df['BMI'] == 0).count() 
sns.histplot(data=df['BMI']) 
df.dtypes.value_counts().plot(kind='bar') 
import copy
df_md = copy.deepcopy(df) 
mode1 = df_md['Glucose'].mode() mean1 = int(df_md['Glucose'].mean()) print('mode:', mode1)
print('mean :', mean1)
df_md['Glucose'] = df_md['Glucose'].replace(to_replace=[0, 0], value=mode1) 
mode2 = df_md['BloodPressure'].mode() mean2 = int(df_md['BloodPressure'].mean()) print('mode:', mode2)
print('mean :', mean2)
df_md['BloodPressure'] = df_md['BloodPressure'].replace(to_replace=[0], value=mode2) 
mean3 = int(df_md['SkinThickness'].mean()) mode3 = df_md['SkinThickness'].mode() print('mode:', mode3)
print('mean :', mean3)
df_md['SkinThickness'] = df_md['SkinThickness'].replace(to_replace=[0], value=mean3) 
mode4 = df_md['Insulin'].mode() mean4 = int(df_md['Insulin'].mean()) print('mode:', mode4)
print('mean :', mean4)
df_md['Insulin'] = df_md['Insulin'].replace(to_replace=[0], value=mean4) 
mode5 = df_md['BMI'].mode() mean5 = int(df_md['BMI'].mean()) print('mode:', mode5) print('mean :', mean5)
df_md['BMI'] = df_md['BMI'].replace(to_replace=[0], value=mean5) 
Positive = df_md[df_md['Outcome']==1]
Positive.groupby('Outcome').hist(figsize=(14, 13),histtype='stepfilled',bins=20,color="green",edgecolor="red")
Negative = df_md[df_md['Outcome']==0]
Negative.groupby('Outcome').hist(figsize=(14, 13),histtype='stepfilled',bins=20,color="red",edgecolor="green")
sns.scatterplot(x= "BloodPressure" ,y= "Glucose", hue="Outcome",data=df_md) 
sns.scatterplot(x= "BMI" ,y= "Insulin",hue="Outcome", data=df_md) 
sns.scatterplot(x= "SkinThickness" ,y= "Insulin", hue="Outcome", data=df_md) 
sns.pairplot(df_md, hue='Outcome') 
df_md.corr()
plt.subplots(figsize=(10,10)) sns.heatmap(df_md.corr(),annot=True,cmap='viridis') 
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = df_md[feature_names] y = df_md.Outcome X.head()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state =10) 
from sklearn.linear_model import LogisticRegression from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score 
accuracyScores = [] modelScores = [] models = []
names = []
models.append(('LR', LogisticRegression(solver='liblinear'))) models.append(('SVC', SVC()))
models.append(('KNN', KNeighborsClassifier())) models.append(('DT', DecisionTreeClassifier())) models.append(('GNB', GaussianNB())) models.append(('RF', RandomForestClassifier())) models.append(('GB', GradientBoostingClassifier())) 
for name, model in models:
model.fit(X_train, y_train) modelScores.append(model.score(X_train,y_train)) y_pred = model.predict(X_test) accuracyScores.append(accuracy_score(y_test, y_pred)) names.append(name)
tr_split_data = pd.DataFrame({'Name': names, 'Score': modelScores,'Accuracy Score': accuracyScores})
print(tr_split_data) 
plt.subplots(figsize=(12,7))
axis = sns.barplot(x = 'Name', y = 'Accuracy Score', data = tr_split_data) axis.set(xlabel='Classifier Name', ylabel='Accuracy Score')
for p in axis.patches: height = p.get_height()
axis.text(p.get_x() + p.get_width()/2, height + 0.005, '{:1.3f}'.format(height), ha="center") plt.show()
names = []
scores = []
for name, model in models: kfold = KFold(n_splits=10)
score = cross_val_score(model, X, y, cv=kfold, scoring='accuracy').mean() names.append(name)
scores.append(score)
k_fold_cross_val_score = pd.DataFrame({'Name': names, 'Score': scores}) print(k_fold_cross_val_score)
plt.subplots(figsize=(12,7))
axis = sns.barplot(x = 'Name', y = 'Score', data = k_fold_cross_val_score) axis.set(xlabel='Classifier Name', ylabel='Accuracy Score')
for p in axis.patches: height = p.get_height()
axis.text(p.get_x() + p.get_width()/2, height + 0.005, '{:1.3f}'.format(height), ha="center") plt.show()
models = [] names = []
models.append(('LR', LogisticRegression(solver='liblinear'))) models.append(('SVC', SVC()))
models.append(('KNN', KNeighborsClassifier())) models.append(('DT', DecisionTreeClassifier())) models.append(('GNB', GaussianNB())) models.append(('RF', RandomForestClassifier())) models.append(('GB', GradientBoostingClassifier())) 
from sklearn.metrics import classification_report, f1_score, accuracy_score, mean_squared_error, roc_auc_score, confusion_matrix, roc_curve, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler 
def report(): auc = [] recall = [] precision = []
specificity = [] names = [] clas = []
for name, model in models:
model.fit(X_train, y_train) y_pred = model.predict(X_test)
auc.append(roc_auc_score(y_test, y_pred)) recall.append(recall_score(y_test, y_pred)) precision.append(precision_score(y_test, y_pred)) cm = (confusion_matrix(y_test, y_pred)) specificity.append((cm[0][0]/(cm[0][1]+cm[0][0]))) names.append(name)
tr_spt_data = pd.DataFrame({'Name': names, 'auc': auc,'Sensitivity': recall,'Specificity' : specificity, 'precision' : precision})
print(tr_spt_data) print('**'*27)
for name, model in models: model.fit(X_train, y_train) y_pred = model.predict(X_test) print('\n')
print('\t\t\tName :', name) print(classification_report(y_test, y_pred)) print('--'*27)

report()

