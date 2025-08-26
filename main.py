from sklearn.model_selection import GridSearchCV
import sklearn.ensemble
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('House_Rent_Dataset.csv')

df['Size'] = pd.to_numeric(df['Size'], errors='coerce')
df = df.dropna(subset=['Size', 'Rent', 'Bathroom', 'BHK'])

X = df[['Rent', 'Bathroom', 'BHK']]
Y = df['Size']

df['Rent'] = np.log1p(df['Rent'])
df['Size'] = np.log1p(df['Size'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, Y_train)

predictions = model.predict(X_test_scaled)

aram_grid = {
  'n_estimators': [100, 200, 500],
  'max_depth': [None, 10, 20, 30],
  'min_samples_split': [2, 5, 10],
}

grid_search = GridSearchCV(RandomForestRegressor(), aram_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, Y_train)

print("Best parameters:", grid_search.best_params_)
print("Best score:", -grid_search.best_score_)

r2 = r2_score(Y_test, predictions)
mae = mean_absolute_error(Y_test, predictions)
mse = mean_squared_error(Y_test, predictions)
rmse = np.sqrt(mse)

print("R²:", r2)
print("MAE:", mae)
print("RMSE:", rmse)
print("MSE:", mse)



plt.scatter(X_test['Rent'], predictions, color='red', label='Model Predictions', alpha=0.5)
plt.xlabel('Rent')
plt.ylabel('Size')
plt.title('Linear Regression Model (with Feature Scaling)')
plt.legend()
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y):,}'))
plt.show()

bins = [0, 5000, 10000, 20000, 50000, 100000, 200000]
labels = ["<5k", "5k-10k", "10k-20k", "20k-50k", "50k-100k", "100k-200k"]
df['Rent_bin'] = pd.cut(df['Rent'], bins=bins, labels=labels, include_lowest=True)
avg_size = df.groupby('Rent_bin')['Size'].mean()
avg_size.plot(kind='bar', color='skyblue', edgecolor='black')
plt.ylabel("Average Size (sq ft)")
plt.xlabel("Rent Range")
plt.title("Average House Size by Rent Range")
plt.xticks(rotation=45)
plt.show()

