import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, roc_auc_score,
    classification_report
)

# ══════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════
print("Loading features_train.csv...")
df = pd.read_csv("output/features_train.csv").dropna()

exclude  = ['label', 'video', 'frame', 'scenario']
FEATURES = [c for c in df.columns if c not in exclude]

X = df[FEATURES].values
y = df['label'].values

print(f"Total frames : {len(df):,}")
print(f"Features     : {len(FEATURES)}  →  {FEATURES}")
print(f"Alert (0)    : {(y==0).sum():,}   Drowsy (1): {(y==1).sum():,}")

# ══════════════════════════════════════════════════════════════════
# 2. SPLIT  (80 / 20, stratified)
# ══════════════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTrain : {len(X_train):,}   Test : {len(X_test):,}")

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ══════════════════════════════════════════════════════════════════
# 3. MODELS  (your intended comparison set)
# ══════════════════════════════════════════════════════════════════
MODELS = {
    "Random Forest (Ours)": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        max_samples=0.8,
        n_jobs=-1,
        random_state=42
    ),
    "MLP Neural Net": MLPClassifier(
        hidden_layer_sizes=(100, 50, 25),
        max_iter=500,
        random_state=42
    ),
    "SVM (RBF)": SVC(
        kernel='rbf',
        probability=True,
        random_state=42
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),
}

# ══════════════════════════════════════════════════════════════════
# 4. TRAIN + EVALUATE ALL MODELS
# ══════════════════════════════════════════════════════════════════
results        = {}
best_acc       = 0.0
best_model_name = ""

print(f"\n{'Algorithm':<22} | {'Acc':>7} | {'Prec':>7} | {'Rec':>7} | {'F1':>7} | {'AUC':>7}")
print("-" * 70)

for name, clf in MODELS.items():
    print(f"  Training {name}...", flush=True)
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)

    acc  = accuracy_score(y_test, y_pred)  * 100
    prec = precision_score(y_test, y_pred, zero_division=0) * 100
    rec  = recall_score(y_test, y_pred,    zero_division=0) * 100
    f1   = f1_score(y_test, y_pred,        zero_division=0) * 100

    try:
        auc = roc_auc_score(y_test, clf.predict_proba(X_test_s)[:, 1]) * 100
    except Exception:
        auc = 0.0

    results[name] = {
        "acc": acc, "prec": prec, "rec": rec,
        "f1": f1,   "auc": auc,  "y_pred": y_pred
    }

    print(f"{name:<22} | {acc:>6.2f}% | {prec:>6.2f}% | {rec:>6.2f}% | {f1:>6.2f}% | {auc:>6.2f}%")

    if acc > best_acc:
        best_acc        = acc
        best_model_name = name
        joblib.dump({
            "clf"       : clf,
            "scaler"    : scaler,
            "features"  : FEATURES,
            "model_name": name          # ← key track.py expects
        }, "drowsy_classifier.pkl")

print(f"\nBest model : {best_model_name}  ({best_acc:.2f}%)")
print(f"Saved      : drowsy_classifier.pkl")

# Full sklearn report for best model
print(f"\n=== {best_model_name} — Classification Report ===")
print(classification_report(
    y_test, results[best_model_name]["y_pred"],
    target_names=["Alert", "Drowsy"]
))

# ══════════════════════════════════════════════════════════════════
# 5. FIGURE 1 — Accuracy comparison bar chart  (train2 style)
# ══════════════════════════════════════════════════════════════════
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
names_list  = list(results.keys())
acc_values  = [results[n]["acc"] for n in names_list]

bar_plot = sns.barplot(x=names_list, y=acc_values, palette="viridis")
plt.axhline(90.48, color='red', linestyle='--',
            label="Adaptive-ECR (Lamba et al. 2025) Baseline (90.48%)")

for i, p in enumerate(bar_plot.patches):
    bar_plot.annotate(
        f'{acc_values[i]:.2f}%',
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center', va='center',
        xytext=(0, 9), textcoords='offset points',
        fontweight='bold'
    )

plt.ylim(75, 100)
plt.ylabel("Accuracy (%)", fontweight='bold')
plt.title("Comparative Analysis: Proposed System vs. Benchmarks", fontweight='bold')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("classifier_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: classifier_comparison.png")

# ══════════════════════════════════════════════════════════════════
# 6. FIGURE 2 — Confusion Matrix  (train_.py style)
#    Shows: raw count + row-% in each cell
#    Title : model name, Accuracy, F1, AUC-ROC
#    Axes  : styled heatmap with colour-bar off
# ══════════════════════════════════════════════════════════════════
print(f"\nGenerating confusion matrix for {best_model_name}...")

best_y_pred = results[best_model_name]["y_pred"]
best_f1     = results[best_model_name]["f1"]
best_auc    = results[best_model_name]["auc"]

cm      = confusion_matrix(y_test, best_y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100   # row %

# Cell annotation: "1234\n(56.78%)"
cell_labels = np.array([
    "{:d}\n({:.2f}%)".format(count, perc)
    for count, perc in zip(cm.flatten(), cm_norm.flatten())
]).reshape(2, 2)

fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor("#F4F6F9")
ax.set_facecolor("#F4F6F9")

sns.heatmap(
    cm,
    annot=cell_labels,
    fmt="",
    cmap="Blues",
    xticklabels=["Predicted: Alert", "Predicted: Drowsy"],
    yticklabels=["True: Alert",      "True: Drowsy"],
    linewidths=1.5,
    annot_kws={"size": 14, "weight": "bold"},
    cbar=False,
    ax=ax
)

ax.set_title(
    f"Confusion Matrix — NTHU-DDD Dataset\n"
    f"{best_model_name} | Accuracy: {best_acc:.2f}%   F1: {best_f1:.2f}%   AUC-ROC: {best_auc:.2f}%",
    fontsize=11, fontweight="bold"
)

plt.tight_layout()
plt.savefig("confusion_matrix_final.png", dpi=300,
            bbox_inches='tight', facecolor="#F4F6F9")
plt.close()
print("Saved: confusion_matrix_final.png")

# ══════════════════════════════════════════════════════════════════
# 7. EXPORT SUMMARY TEXT
# ══════════════════════════════════════════════════════════════════
with open("evaluation_results.txt", "w", encoding="utf-8") as f:
    f.write("=" * 50 + "\n")
    f.write("DROWSINESS DETECTION — EVALUATION RESULTS\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Best model  : {best_model_name}\n")
    f.write(f"Accuracy    : {results[best_model_name]['acc']:.2f}%\n")
    f.write(f"Precision   : {results[best_model_name]['prec']:.2f}%\n")
    f.write(f"Recall      : {results[best_model_name]['rec']:.2f}%\n")
    f.write(f"F1-Score    : {results[best_model_name]['f1']:.2f}%\n")
    f.write(f"AUC-ROC     : {results[best_model_name]['auc']:.2f}%\n\n")
    f.write("All models:\n")
    for n, r in results.items():
        marker = "" if n == best_model_name else ""
        f.write(f"  {n:<22}  Acc={r['acc']:.2f}%  F1={r['f1']:.2f}%{marker}\n")
    f.write("\n" + "=" * 50 + "\n")

print("Saved: final_metrics.txt")

print(f"\n{'='*50}")
print("  DONE")
print(f"{'='*50}")
print(f"  Winner  : {best_model_name} ({best_acc:.2f}%)")
print(f"  Outputs : drowsy_classifier.pkl")
print(f"            classifier_comparison.png")
print(f"            confusion_matrix_final.png")
print(f"            final_metrics.txt")
print(f"{'='*50}")