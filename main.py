import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1. Generate Synthetic Dataset (100 Rows)
np.random.seed(101)
n_samples = 100

area = np.random.randint(500, 3500, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
age = np.random.randint(1, 15, n_samples)
location_scores = np.random.uniform(1.0, 10.0, n_samples)
city_dist = np.random.uniform(1.0, 30.0, n_samples)

# Price logic
prices = (area * 0.04) + (bedrooms * 5.0) + (location_scores * 3.0) - (age * 0.5) - (city_dist * 0.7) + np.random.normal(0, 5, n_samples)

df = pd.DataFrame({
    'Area_SqFt': area,
    'Bedrooms': bedrooms,
    'Age_Years': age,
    'Location_Score': np.round(location_scores, 2),
    'Distance_City_KM': np.round(city_dist, 2),
    'Price_Lakhs': np.round(prices, 2)
})

# Save to CSV
df.to_csv('dataset.csv', index=False)

# 2. Machine Learning Model Training
X = df[['Area_SqFt', 'Bedrooms', 'Age_Years', 'Location_Score', 'Distance_City_KM']]
y = df['Price_Lakhs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# 3. Model Evaluation
print("SUCCESS: dataset.csv created with 100 rows!")
print("Model trained and evaluated successfully.")
print("Model Score (R2):", round(model.score(X_test, y_test), 2))
