import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import FunctionTransformer, Pipeline

# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------
df = pd.read_csv('./assets/Dataset3.csv', sep=';')

X = df.drop(['ID', 'default.payment.next.month'], axis=1)
y = df['default.payment.next.month']

dummy_cols = ['SEX', 'EDUCATION', 'MARRIAGE']  # categorical features
pay_cols = [f'PAY_{i}' for i in [0, 2, 3, 4, 5, 6]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ------------------------------------------------------------
# Definition of preprocessing pipelines
# ------------------------------------------------------------

skew_feats = [f'BILL_AMT{i}' for i in range(1, 7)] + \
             [f'PAY_AMT{i}'  for i in range(1, 7)]

# Pipeline: impute 0 → log1p → RobustScaler
skew_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('log', FunctionTransformer(lambda X: np.sign(X)*np.log1p(np.abs(X)), feature_names_out='one-to-one')),
    ('rscale', RobustScaler())
])

ordinal_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', StandardScaler())
])

# ------------------------------------------------------------
# Evaluation functions
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

def evaluate_balanced(model, X, y, label="Test"):
    """
    Stampa Balanced-Accuracy e PR-AUC (media precision-recall area).
    """
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    bal_acc = balanced_accuracy_score(y, y_pred)
    pr_auc  = average_precision_score(y, y_proba)

    print(f"[{label}] Balanced-Acc={bal_acc:.3f} | PR-AUC={pr_auc:.3f}")

# ------------------------------------------------------------
# Complete pipeline: Random Forest with all features
# ------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Feature definition
categorical_features = dummy_cols
numeric_features = [c for c in X_train.columns if c not in categorical_features]

# Subset of numeric features that are not skewed that will be standardized
numeric_other = [c for c in numeric_features if c not in skew_feats + pay_cols]


preprocessor = ColumnTransformer([
    ('skew', skew_pipe, skew_feats),
    ('num', StandardScaler(), numeric_other),
    ('ord', ordinal_pipe, pay_cols),
    ('cat', OneHotEncoder(
        drop='first',
        handle_unknown='ignore',
        sparse_output=False,
        min_frequency=2
    ), categorical_features)
])

rf_base = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
pipe_rf = Pipeline([
    ('prep', preprocessor),
    ('rf', rf_base)
])

param_dist = {
    "rf__n_estimators": list(range(400, 1001, 100)),
    "rf__max_depth":    [None, 15, 20, 25, 30],
    "rf__max_features": ["sqrt", "log2", 0.5, None],
    "rf__min_samples_split": [2, 5, 10],
    "rf__min_samples_leaf":  [5, 10, 20],
    "rf__class_weight":  ["balanced", "balanced_subsample"],
    "rf__max_samples":   [None, 0.8, 0.6]
}

rf_search = RandomizedSearchCV(
    estimator=pipe_rf,
    param_distributions=param_dist,
    n_iter=50,
    scoring='average_precision',
    cv=cv,
    n_jobs=-1,
    random_state=42,
    verbose=1
)

rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_

evaluate(best_rf, X_train, X_test, y_train, y_test,
         label="Random Forest - TUTTE le feature")
evaluate_balanced(best_rf, X_test, y_test)

# ------------------------------------------------------------
# Complete pipeline: Random Forest with Lasso feature selection
# ------------------------------------------------------------

pipe_lasso_rf = Pipeline([
    ('prep', preprocessor),
    ('sel',  SelectFromModel(
                 LogisticRegression(penalty='l1',
                                    solver='saga',
                                    class_weight='balanced',
                                    max_iter=5000),
                 threshold='median',
                 max_features=30)),
    ('rf',   rf_base)
])

param_grid_lasso_rf = {
    # Lasso
    'sel__estimator__C':  np.logspace(-2, 1, 6),
    # Random Forest
    'rf__n_estimators':   [600, 800],
    'rf__max_depth':      [15, 20, 25],
    'rf__min_samples_leaf': [5, 10],
    'rf__max_features':   ['sqrt', 0.5]
}

search_lasso_rf = GridSearchCV(
    pipe_lasso_rf,
    param_grid_lasso_rf,
    scoring='average_precision',
    cv=cv,
    n_jobs=-1,
    verbose=1
)

search_lasso_rf.fit(X_train, y_train)
best_rf_sel = search_lasso_rf.best_estimator_

sel_step   = best_rf_sel.named_steps['sel']
prep_step  = best_rf_sel.named_steps['prep']

mask = sel_step.get_support()
feat = prep_step.get_feature_names_out()
print("Feature tenute:", feat[mask])

evaluate(best_rf_sel, X_train, X_test, y_train, y_test,
         label="Random Forest - dopo Lasso")
evaluate_balanced(best_rf_sel, X_test, y_test)
