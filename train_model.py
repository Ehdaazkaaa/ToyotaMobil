
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
import joblib

# Load data
df = pd.read_csv("Toyota (1).csv")

# Preprocessing
le = LabelEncoder()
df['model_encoded'] = le.fit_transform(df['model'])

# Fitur & target
features = ['model_encoded', 'year', 'mileage', 'tax', 'mpg', 'engineSize']
X = df[features]
y = df['price']

# Melatih model
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X, y)

# Simpan model & encoder
joblib.dump(knn, 'knn_model.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("✅ Model dan encoder berhasil disimpan.")
