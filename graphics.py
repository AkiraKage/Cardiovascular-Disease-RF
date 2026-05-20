import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc

cartella_output = "graphic"
if not os.path.exists(cartella_output):
    os.makedirs(cartella_output)

print("Generazione dei grafici...")
# caricamento del modello e dei dati di test
pipeline_data = joblib.load("cardio_rf.joblib")
rf_model = pipeline_data["model"]
X_test = pipeline_data["X_test"]
y_test = pipeline_data["y_test"]
feature_names = pipeline_data["feature_names"]

soglia_medica = 0.42
y_prob = rf_model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= soglia_medica).astype(int)

sns.set_theme(style="whitegrid")

# matrice di confusione
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["Sano (0)", "Malato (1)"],
    yticklabels=["Sano (0)", "Malato (1)"],
    annot_kws={"size": 14},
)
plt.title("Matrice di Confusione", fontsize=16, pad=15, fontweight="bold")
plt.xlabel("Previsione del Modello")
plt.ylabel("Stato Reale")
plt.tight_layout()
plt.savefig(os.path.join(cartella_output, "1_matrice_confusione.png"), dpi=300)
plt.close()

# matrice di correlazione
dati_corr = X_test.copy()
dati_corr["cardio"] = y_test
matrice_correlazione = dati_corr.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(
    matrice_correlazione,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar=True,
    square=True,
    linewidths=0.5,
)
plt.title(
    "Matrice di Correlazione tra Variabili Cliniche",
    fontsize=16,
    pad=20,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(os.path.join(cartella_output, "2_matrice_correlazione.png"), dpi=300)
plt.close()

# curva ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2.5, label=f"ROC (Area = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("Tasso Falsi Positivi")
plt.ylabel("Tasso Veri Positivi")
plt.title("Curva ROC", fontsize=16, pad=15, fontweight="bold")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(cartella_output, "3_curva_roc.png"), dpi=300)
plt.close()

# distribuzione probabilità
plt.figure(figsize=(10, 6))
sns.histplot(
    y_prob[y_test == 0],
    color="mediumseagreen",
    label="Sani Reali",
    kde=True,
    stat="density",
    bins=40,
    alpha=0.6,
)
sns.histplot(
    y_prob[y_test == 1],
    color="crimson",
    label="Malati Reali",
    kde=True,
    stat="density",
    bins=40,
    alpha=0.6,
)
plt.title(
    "Distribuzione delle Probabilità Cardiovascolari",
    fontsize=16,
    pad=15,
    fontweight="bold",
)
plt.xlabel("Probabilità calcolata di avere una malattia cardiovascolare", fontsize=12)
plt.ylabel("Densità")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(cartella_output, "4_distribuzione_probabilita.png"), dpi=300)
plt.close()

# feature importance
feature_importances = pd.DataFrame(
    {"Variabile": feature_names, "Importanza": rf_model.feature_importances_}
).sort_values(by="Importanza", ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(
    x="Importanza",
    y="Variabile",
    data=feature_importances,
    hue="Variabile",
    palette="viridis",
    legend=False,
)
plt.title("Fattori di Rischio più Rilevanti", fontsize=16, pad=20, fontweight="bold")
plt.xlabel("Peso Decisionale")
plt.tight_layout()
plt.savefig(os.path.join(cartella_output, "5_feature_importance.png"), dpi=300)
plt.close()

print(f"Grafici salvati con successo in '{cartella_output}/'.")
