# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# %%
file = "bmw_global_sales_2018_2025.csv"
data =pd.read_csv(file)

# %%
pd.set_option("display.max_rows",None)


# %%
print(data.columns)

# %%
plt.figure(figsize =(10,5))
sns.barplot(x="Year",y="Units_Sold",data=data,hue="Model")
plt.show()

# %%
plt.figure(figsize=(10,5))
sns.barplot(x="Region",y="GDP_Growth",data=data,hue="Year")
plt.show()

# %%
plt.figure(figsize=(10,5))
sns.barplot(x="Model",y="Units_Sold",data =data,hue="Region")
plt.show()

# %%
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

encoder = OneHotEncoder(sparse_output=False)

for col in data.select_dtypes(include="object"):
    
    encoded = encoder.fit_transform(data[[col]])
    
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out([col]),
        index=data.index
    )
    
    data = pd.concat([data.drop(col, axis=1), encoded_df], axis=1)

    

    

# %%
print(data.head(5))

# %%
corr = data.corr()["Units_Sold"].sort_values(ascending=True)
print(corr)

# %%
plt.figure(figsize=(10,5))
sns.heatmap(data=data.corr())
plt.show()

# %%
y =data["Units_Sold"]
X= data.drop(["Units_Sold"],axis=1)


# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
LR =LinearRegression()
LR.fit(X_train,y_train)
prediction =LR.predict(X_test)
print(f"R2 : {round(r2_score(y_test,prediction),2)}")
print(f"MSE: {round(mean_squared_error(y_test,prediction),2)}")
print(f"MAE: {round(mean_absolute_error(y_test,prediction),2)}")



R2 : 0.95
MSE: 474997.85
MAE: 481.68

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
LR =RandomForestRegressor(n_estimators=200,random_state=42)
LR.fit(X_train,y_train)
prediction =LR.predict(X_test)
print(f"R2 : {round(r2_score(y_test,prediction),2)}")
print(f"MSE: {round(mean_squared_error(y_test,prediction),2)}")
print(f"MAE: {round(mean_absolute_error(y_test,prediction),2)}")

R2 : 1.0
MSE: 21070.71
MAE: 91.39
