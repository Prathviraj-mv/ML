# %%
import pandas as pd

file ="student_lifestyle_100k.csv"
data = pd.read_csv(file)
pd.set_option("display.max_rows",None)


# %%
print(data.isnull().sum())

# %%
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,15))
sns.stripplot(data=data,x ="Department",y="CGPA",hue="Depression")
plt.show()

# %%
plt.figure(figsize=(10,15))
sns.boxplot(data=data,x ="Department",y="CGPA",hue="Depression")
plt.show()

# %%
plt.figure(figsize=(10,15))
sns.jointplot(data=data,x ="Sleep_Duration",y="Study_Hours",hue="Depression")
plt.show()

# %%
data["study_sleep"] = data["Sleep_Duration"]/data["Study_Hours"]



# %%
plt.figure(figsize=(10,15))
sns.jointplot(data=data,x ="Sleep_Duration",y="CGPA",hue="Depression")
plt.show()

# %%
plt.figure(figsize=(10,15))
sns.jointplot(data=data,x ="Social_Media_Hours",y="CGPA",hue="Depression")
plt.show()

# %%
from sklearn.preprocessing import LabelEncoder

for col in data.select_dtypes(include="object"):
    lE = LabelEncoder()
    data[col] =lE.fit_transform(data[col])

corr = data.corr()["Depression"].sort_values(ascending=True)
print(corr)

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# %%
X = data.drop(
    columns=["Student_ID", "Age", "Gender", "Department", "Depression","study_sleep"],
    errors="ignore"
)

y = data["Depression"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
RF =RandomForestClassifier(n_estimators=200)

RF.fit(X_train,y_train)
pred = RF.predict(X_test)

print(classification_report(y_pred=pred,y_true=y_test))
print(confusion_matrix(y_pred=pred,y_true=y_test))
print(round(accuracy_score(y_pred=pred,y_true=y_test),2))
print(round(precision_score(y_pred=pred,y_true=y_test),2))

# precision    recall  f1-score   support

#        False       0.90      1.00      0.95     29694
#         True       0.45      0.03      0.05      3306

#     accuracy                           0.90     33000
#    macro avg       0.68      0.51      0.50     33000
# weighted avg       0.86      0.90      0.86     33000

# [[29577   117]
#  [ 3211    95]]
# acc 0.9
# prec 0.45

# %%

pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),("model", RandomForestClassifier(
                                                 n_estimators=200,
                                                class_weight="balanced"))

    ]
)

param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 8, 12]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    scoring="recall",
    cv=3,
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train,y_train)
pred = grid.predict(X_test)

print(classification_report(y_pred=pred,y_true=y_test))
print(confusion_matrix(y_pred=pred,y_true=y_test))
print(round(accuracy_score(y_pred=pred,y_true=y_test),2))
print(round(precision_score(y_pred=pred,y_true=y_test),2))

# Fitting 3 folds for each of 9 candidates, totalling 27 fits
#               precision    recall  f1-score   support

#        False       0.95      0.75      0.83     29694
#         True       0.21      0.62      0.32      3306

#     accuracy                           0.73     33000
#    macro avg       0.58      0.69      0.58     33000
# weighted avg       0.87      0.73      0.78     33000

# [[22156  7538]
#  [ 1243  2063]]
# acc 0.73
# prec 0.21
# %%
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
xgb_model = xgb.XGBClassifier(
    tree_method="hist",
    device="cuda",
    objective="binary:logistic",
    eval_metric="logloss"
)

param_grid = {
    "n_estimators": [200, 300],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

grid = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring="accuracy",
    cv=3,
    verbose=2,
    n_jobs=1
)

grid.fit(X_train,y_train)
pred = grid.predict(X_test)

print(classification_report(y_pred=pred,y_true=y_test))
print(confusion_matrix(y_pred=pred,y_true=y_test))
print(round(accuracy_score(y_pred=pred,y_true=y_test),2))
print(round(precision_score(y_pred=pred,y_true=y_test),2))

#               precision    recall  f1-score   support

#        False       0.90      1.00      0.95     29694
#         True       0.47      0.02      0.05      3306

#     accuracy                           0.90     33000
#    macro avg       0.69      0.51      0.50     33000
# weighted avg       0.86      0.90      0.86     33000

# [[29605    89]
#  [ 3226    80]]
# acc 0.9
# prec 0.47
