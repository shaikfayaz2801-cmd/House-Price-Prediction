import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

data = pd.read_csv("Housing.csv")

print("First 5 rows:")
print(data.head())


yes_no_columns = [
    'mainroad',
    'guestroom',
    'basement',
    'hotwaterheating',
    'airconditioning',
    'prefarea'
]

for col in yes_no_columns:
    data[col] = data[col].map({'yes': 1, 'no': 0})


data['furnishingstatus'] = data['furnishingstatus'].map({
    'unfurnished': 0,
    'semi-furnished': 1,
    'furnished': 2
})

print("\nUpdated Dataset:")
print(data.head())


X = data.drop('price', axis=1)
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel trained successfully!")

y_pred = model.predict(X_test)


accuracy = r2_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\nMean Absolute Error (MAE):")
print(mae)

print("\nMean Squared Error (MSE):")
print(mse)

print("\nRoot Mean Squared Error (RMSE):")
print(rmse)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")

plt.show()

joblib.dump(model, "house_price_model.pkl")

print("\nModel saved successfully!")
print("house_price_model.pkl file created!")

print("\nEnter house details to predict price")

area = int(input("Enter area: "))
bedrooms = int(input("Enter bedrooms: "))
bathrooms = int(input("Enter bathrooms: "))
stories = int(input("Enter stories: "))
mainroad = int(input("Main road? (1=yes, 0=no): "))
guestroom = int(input("Guest room? (1=yes, 0=no): "))
basement = int(input("Basement? (1=yes, 0=no): "))
hotwaterheating = int(input("Hot water heating? (1=yes, 0=no): "))
airconditioning = int(input("Air conditioning? (1=yes, 0=no): "))
parking = int(input("Parking spaces: "))
prefarea = int(input("Preferred area? (1=yes, 0=no): "))
furnishingstatus = int(input(
    "Furnishing (2=furnished, 1=semi, 0=unfurnished): "
))

# Input for prediction
new_house = [[
    area,
    bedrooms,
    bathrooms,
    stories,
    mainroad,
    guestroom,
    basement,
    hotwaterheating,
    airconditioning,
    parking,
    prefarea,
    furnishingstatus
]]

predicted_price = model.predict(new_house)

print("\nPredicted House Price:")
print(f"₹ {predicted_price[0]:,.2f}")