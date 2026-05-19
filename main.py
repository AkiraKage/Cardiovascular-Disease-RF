import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib

# caricamento e pulizia dati
print("Caricamento e pulizia dati in corso...")
df = pd.read_csv("cardio_train.csv", sep=";")
df = df.drop("id", axis=1)
df["age"] = (df["age"] / 365.25).astype(int)  # conversione età giorni -> anni

# filtraggio valori impossibili della pressione sanguigna
# limiti medici ragionevoli: sistolica (70-240) e diastolica (40-150)
df_clean = df[
    (df["ap_hi"] >= 70)
    & (df["ap_hi"] <= 240)
    & (df["ap_lo"] >= 40)
    & (df["ap_lo"] <= 150)
    & (df["ap_hi"] > df["ap_lo"])
].copy()

print("Calcolo nuove feature cliniche...")
# calcolo BMI (Body Mass Index)
df_clean["bmi"] = df_clean["weight"] / ((df_clean["height"] / 100) ** 2)
df_clean["bmi"] = df_clean["bmi"].round(2)
# calcolo pressione differenziale (pulse pressure)
df_clean["pulse_pressure"] = df_clean["ap_hi"] - df_clean["ap_lo"]

# campionamento di 20.000 record per non far durare la ricerca parametri ore
df_sampled = df_clean.sample(n=20000, random_state=42)

X = df_sampled.drop("cardio", axis=1)
y = df_sampled["cardio"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ottimizzazione iperparametri con RandomizedSearchCV
param_distributions = {
    "n_estimators": [100, 200, 300],
    "max_depth": [8, 10, 15],  # profondità bassa per evitare overfitting
    "min_samples_split": [5, 10, 15],
    "min_samples_leaf": [2, 4, 6],
}

rf_base = RandomForestClassifier(random_state=42, class_weight="balanced")

# addestramento e ricerca iperparametri
print("Inizio dell'addestramento e ottimizzazione...")
random_search = RandomizedSearchCV(
    rf_base,
    param_distributions=param_distributions,
    n_iter=10,
    cv=3,
    scoring="roc_auc",
    random_state=42,
    n_jobs=-1,
    verbose=1,
)

random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

print(f"\nMigliori iperparametri rilevati: {random_search.best_params_}")

# archiviazione del modello e dei dati di test in un unico file
export_data = {
    "model": best_model,
    "X_test": X_test,
    "y_test": y_test,
    "feature_names": X.columns,
}

joblib.dump(export_data, "cardio_rf.joblib")

print("Modello e dati salvati in 'cardio_rf.joblib'.")
