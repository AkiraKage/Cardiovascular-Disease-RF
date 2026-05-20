import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import joblib

print("Caricamento del modello...")
# caricamento dizionario unico
pipeline_data = joblib.load("cardio_rf.joblib")

# estrazione del modello e dei dati di test
rf_model = pipeline_data["model"]
X_test = pipeline_data["X_test"]
y_test = pipeline_data["y_test"]

# previsioni
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
soglia_medica = 0.42
y_pred = (y_pred_proba >= soglia_medica).astype(int)

# metriche
roc_auc = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)
report_class = classification_report(
    y_test, y_pred, target_names=["Sano (0)", "Malato (1)"]
)
matrice_conf = confusion_matrix(y_test, y_pred)

print(f"ROC AUC Score: {roc_auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# creazione e salvataggio del report riassuntivo in txt
testo_report = f"""REPORT DI VALUTAZIONE - MALATTIE CARDIOVASCOLARI\n
Metriche di Performance:
ROC-AUC Score: {roc_auc:.4f}
Accuratezza: {accuracy:.4f}

REPORT DI CLASSIFICAZIONE:
{report_class}

MATRICE DI CONFUSIONE:
{matrice_conf}
[Riga 1: Veri Sani, Falsi Malati]
[Riga 2: Falsi Sani, Veri Malati]
"""

with open("report.txt", "w") as f:
    f.write(testo_report)

print("-> Report finale salvato in 'report.txt'")
