import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


np.random.seed(42)
n_samples = 1000

applicant_income = np.random.normal(7000000, 2500000, n_samples).clip(2000000)
loan_amount = np.random.normal(50000000, 20000000, n_samples).clip(5000000)
credit_score = np.random.randint(300, 850, size=n_samples)
employed = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.8, 0.2])


approval_score = (
    (credit_score / 850) * 0.5
    + (applicant_income / (loan_amount * 0.1)) * 0.3
    + (employed == 'Yes').astype(int) * 0.2
    + np.random.normal(0, 0.1, n_samples)
)
loan_status = (approval_score > 0.55).astype(int)

df = pd.DataFrame({
    'income': applicant_income,
    'loan_amount': loan_amount,
    'credit_score': credit_score,
    'employed': employed,
    'loan_status': loan_status,
})


df.loc[np.random.choice(n_samples, 25), 'income'] = np.nan

# Pra-pemrosesan: Tangani missing value & Encoding kategori
df['income'] = df['income'].fillna(df['income'].median())
df['employed'] = df['employed'].map({'No': 0, 'Yes': 1})


X = df.drop(columns=['loan_status'])
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)

print('=== CONFUSION MATRIX ===')
print(confusion_matrix(y_test, y_pred))

print('\n=== CLASSIFICATION REPORT ===')
print(
    classification_report(
        y_test, y_pred, target_names=['Ditolak (0)', 'Disetujui (1)']
    )
)


new_applicants = pd.DataFrame({
    'income': [3000000, 12000000],
    'loan_amount': [80000000, 40000000],
    'credit_score': [450, 780],
    'employed': [0, 1],
})

new_scaled = scaler.transform(new_applicants)
predictions = model.predict(new_scaled)
probabilities = model.predict_proba(new_scaled)[:, 1]

new_applicants['Status_Prediksi'] = [
    'Disetujui' if p == 1 else 'Ditolak' for p in predictions
]
new_applicants['Probabilitas_Lolos'] = probabilities.round(2)

print('\n=== INFERENSI APLIKASI PINJAMAN BARU ===')
print(new_applicants)
