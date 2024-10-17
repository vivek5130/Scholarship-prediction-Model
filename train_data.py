import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
data = pd.read_csv('data/scholarship_data.csv')

# Encode categorical columns (if any)
label_encoder = LabelEncoder()
data['Nationality'] = label_encoder.fit_transform(data['Nationality'])
data['Physically_Disabled'] = label_encoder.fit_transform(data['Physically_Disabled'])
data['Extracurricular'] = label_encoder.fit_transform(data['Extracurricular'])
data['Gender'] = label_encoder.fit_transform(data['Gender'])
data['Caste'] = label_encoder.fit_transform(data['Caste'])

# Separate features (X) and target (y)
X = data.drop(columns=['Scholarship_Eligibility'])
y = data['Scholarship_Eligibility']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model using joblib
joblib.dump(model, 'model/scholarship_model.pkl')

print("Model trained and saved as scholarship_model.pkl")
