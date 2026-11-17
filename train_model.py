import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# 1. Load dataset
data = pd.read_csv("dataset.csv")

# 2. Convert text columns into numbers
data = pd.get_dummies(data, drop_first=True)

# 3. Separate input and output
X = data.drop("charges", axis=1)
y = data["charges"]

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Create ML algorithms
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42
    )
}

results = {}
trained_models = {}

# 6. Train and evaluate models
for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    results[name] = r2
    trained_models[name] = model

    print("\nAlgorithm:", name)
    print("R2 Score:", r2)
    print("Mean Absolute Error:", mae)

# 7. Select best model
best_model_name = max(results, key=results.get)
best_model = trained_models[best_model_name]

print("\nBest Model:", best_model_name)
print("Best R2 Score:", results[best_model_name])

# 8. Save best model and columns
with open("model.pkl", "wb") as file:
    pickle.dump((best_model, X.columns.tolist()), file)

print("\nModel saved successfully!")

# 9. Create model comparison graph
os.makedirs("static", exist_ok=True)

plt.figure(figsize=(9, 5))
plt.bar(results.keys(), results.values())
plt.title("Machine Learning Model Comparison")
plt.xlabel("Algorithms")
plt.ylabel("R2 Score")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig("static/model_comparison.png")
plt.show()

print("\nGraph saved in static/model_comparison.png")