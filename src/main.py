import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

df = pd.read_csv('./assets/Dataset3.csv', sep=';')

X = df.drop(['ID', 'default.payment.next.month'], axis=1)
y = df['default.payment.next.month']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ------------------------------------------------------------
# 2. Funzione di valutazione (riuso per entrambi i modelli)
# ------------------------------------------------------------
def evaluate(model, X_tr, X_te, y_tr, y_te, label):
    print("\n" + "=" * 30, label, "=" * 30)
    for split, (X_split, y_split) in (("Train", (X_tr, y_tr)),
                                      ("Test",  (X_te, y_te))):
        y_pred = model.predict(X_split)
        y_prob = model.predict_proba(X_split)[:, 1]
        print(f"[{split}] "
              f"Accuracy={accuracy_score(y_split, y_pred):.3f} | "
              f"Precision={precision_score(y_split, y_pred):.3f} | "
              f"Recall={recall_score(y_split, y_pred):.3f} | "
              f"F1={f1_score(y_split, y_pred):.3f} | "
              f"AUC={roc_auc_score(y_split, y_prob):.3f}")
    print("\nConfusion matrix (Test):")
    print(confusion_matrix(y_te, model.predict(X_te)))
    print("\nClassification report (Test):")
    print(classification_report(y_te, model.predict(X_te), digits=3))

rf_base = RandomForestClassifier(
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

param_dist = {
    "n_estimators": [int(x) for x in np.linspace(200, 600, 5)],   # 200, 300, …, 600
    "max_depth":    [None, 10, 20, 30],
    "max_features": ["sqrt", "log2", None],
    "min_samples_split": [2, 3, 4],
    "min_samples_leaf":  [1, 2]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

rf_search = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_dist,
    n_iter=5,                      # bastano 5 iterazioni per il compito richiesto
    scoring="roc_auc",
    n_jobs=-1,
    cv=cv,
    random_state=42,
    verbose=1
)

rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_

evaluate(best_rf, X_train, X_test, y_train, y_test,
         label="Random Forest - TUTTE le feature")


# ------------------------------------------------------------
# 4. Lasso (L1) per selezione feature + Random Forest (Modello B)
# ------------------------------------------------------------

scaler = StandardScaler()
lasso_clf = LogisticRegression(
    penalty="l1",
    #C=0.01,                 # forte regolarizzazione → molti coef. a zero
    solver="liblinear",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Pipeline di fit per ottenere i coefficienti L1
pipe_lasso = Pipeline([
    ("scaler", scaler),
    ("lasso", lasso_clf)
])
param_grid_lasso = {
    "lasso__C": [0.001, 0.01, 0.1, 1, 10, 100]
}
search_lasso = GridSearchCV(
    estimator=pipe_lasso,
    param_grid=param_grid_lasso,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

search_lasso.fit(X_train, y_train)

best_C = search_lasso.best_params_["lasso__C"]

lasso_opt = LogisticRegression(
    penalty="l1",
    C=best_C,
    solver="liblinear",
    class_weight="balanced",
    random_state=42,
    max_iter=1000,
    n_jobs=-1
)

# 8. Rifacciamo il fitting per estrarre i coefficienti
pipe_lasso_opt = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", lasso_opt)
])
pipe_lasso_opt.fit(X_train, y_train)

selector = SelectFromModel(
    pipe_lasso_opt.named_steps["lasso"],
    prefit=True,
    threshold="mean"        # mantieni coefficienti sopra la media
)
mask = selector.get_support()
selected_cols = X_train.columns[mask]
print(f"\nFeature selezionate dal Lasso ({mask.sum()} su {X_train.shape[1]}):")
print(list(selected_cols))

# Sub-dataset ridotto
X_train_sel = X_train[selected_cols]
X_test_sel  = X_test[selected_cols]

# Random Forest sul sotto-spazio
rf_search_sel = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_dist,
    n_iter=5,
    scoring="roc_auc",
    n_jobs=-1,
    cv=cv,
    random_state=42,
    verbose=1
)

rf_search_sel.fit(X_train_sel, y_train)
best_rf_sel = rf_search_sel.best_estimator_

evaluate(best_rf_sel, X_train_sel, X_test_sel, y_train, y_test,
         label="Random Forest - dopo Lasso")