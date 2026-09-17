import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
from sklearn.utils import resample

#classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#--------------------------------------------------------------------------------------------------------------------- reused functions and data prep

def imputer(strat :str):
    # strategies: 
    return SimpleImputer(strategy= strat, add_indicator=False)

data = pd.read_csv(r"C:\Users\Nutzer\Desktop\scientific_project\feature_set_generation\output_data\feature_table.csv", sep = ",")

# X: feature rows
# y: target values

# bring in chonological order

data["onset"] = pd.to_datetime(data["onset"])

data = data.sort_values("onset").reset_index(drop =True)

#print("overall",data["class"].sum())

#print(data.head()) check if sorted and it works 

split_idx = int(len(data) * 0.8)

train_data = data.iloc[:split_idx]

test_data = data.iloc[split_idx:]

#print(test_data.head(15))


#print(f"Train-Set Size: {len(train_data)} (till {train_data['onset'].max().strftime('%Y-%m-%d')})")  #works
#print(f"Test-Set Size: {len(test_data)} (from {test_data['onset'].min().strftime('%Y-%m-%d')})")

#--- prep


drop_cols = ["tkr", "onset", "duration", 
              "class", "id","p-onset"]

X_train = train_data.drop(drop_cols, axis = 1)
y_train = train_data["class"].values

#print("train", y_train.sum())

X_test = test_data.drop(drop_cols, axis = 1) 
y_test = test_data["class"].values 
#print("test", y_test.sum())
feature_names = X_train.columns.to_numpy() 

# --------------------------------------------------------------------------- models
models = [
    ("RF", RandomForestClassifier(
        n_estimators=500, max_depth=2, random_state=42), True),
    ("XGB", XGBClassifier(
        max_depth=2, learning_rate=0.05, n_estimators=500, random_state=42), True),
    ("SVC-RBF",SVC(kernel = "rbf", probability= True, random_state=42),True),
    ("LogReg_en", LogisticRegression(solver = "saga", l1_ratio= 0.5 ,C = 1.0,max_iter=1000), True),
    ("NB", GaussianNB(), True),
]



# --------------------------------------------------------------------------- #model trainings loop, unimportant dont forget do depreceiate

test_results = []
feature_importance = []
fitted_pipelines = {}

for name, mdl, needs_proba in models:

    pipe = Pipeline([("impute", imputer(strat = "median")),
                      ("scaler", StandardScaler()),
                      ("model", mdl)])

    pipe.fit(X_train,y_train)

    pred = pipe.predict(X_test)

    fitted_pipelines[name] = pipe 

    auc = np.nan
    if needs_proba and hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)

    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    test_results.append({
        "model" : name,
        "auc": auc,
        "f1" : f1_score(y_test, pred, average= "macro"),
        "bal_acc" : balanced_accuracy_score(y_test, pred),
        "True Neg": tn,
        "False Pos": fp,
        "False Neg": fn,
        "True Pos": tp
    })

test_summary = pd.DataFrame(test_results).set_index("model")




#test_summary.to_csv("model_performance.csv")               #save simple performance (not used in paper, to onedimensional without bootstrapping)



#print(test_summary)


# ---------------------------------------------------------------- configs
N_BOOT = 1000
rng = np.random.default_rng(42)
idx, dropped = [], 0


while len(idx) < N_BOOT:
    b = rng.integers(0, len(y_test), len(y_test))
    if len(np.unique(y_test[b])) > 1:
        idx.append(b)
    else:
        dropped += 1
idx = np.array(idx)

# ---------------------------------------------------------------- metrics
def macro_f1(yt, yp):
    return f1_score(yt, yp, average="macro", zero_division=0)

POINT_METRICS = {"bal_acc": balanced_accuracy_score, "f1": macro_f1}

def ranking_score(pipe, X):
    for attr in ("decision_function", "predict_proba"):
        if hasattr(pipe, attr):
            s = getattr(pipe, attr)(X)
            s = s[:, 1] if s.ndim > 1 else s
            return None if np.ptp(s) == 0 else s
    return None

# ---------------------------------------------------------------- main perofrmance table 
preds, scores, rows = {}, {}, []

for name, pipe in fitted_pipelines.items():
    preds[name] = pipe.predict(X_test)
    scores[name] = ranking_score(pipe, X_test)

    row = {"model": name}
    for key, fn in POINT_METRICS.items():
        vals = np.array([fn(y_test[b], preds[name][b]) for b in idx])
        row[key] = fn(y_test, preds[name])
        row[f"{key}_lo"], row[f"{key}_hi"] = np.percentile(vals, [2.5, 97.5])

    if scores[name] is None:
        row |= {"auc": np.nan, "auc_lo": np.nan, "auc_hi": np.nan}
    else:
        vals = np.array([roc_auc_score(y_test[b], scores[name][b]) for b in idx])
        row |= {"auc": roc_auc_score(y_test, scores[name]),
                "auc_lo": np.percentile(vals, 2.5),
                "auc_hi": np.percentile(vals, 97.5)}
    rows.append(row)

COLS = ["auc", "auc_lo", "auc_hi",
        "bal_acc", "bal_acc_lo", "bal_acc_hi",
        "f1", "f1_lo", "f1_hi"]
df_ci = pd.DataFrame(rows).set_index("model")[COLS].round(3)

# ---------------------------------------------------------------- stats-testing

p_train = y_train.mean()
chance_f1 = np.array([macro_f1(y_test, (rng.random(len(y_test)) < p_train).astype(int))
                      for _ in range(10_000)])
CHANCE = {"auc": 0.5, "bal_acc": 0.5, "f1": chance_f1.mean()}

for key in POINT_METRICS:
    df_ci[f"{key}_p"] = [
        np.mean(np.array([POINT_METRICS[key](y_test[b], preds[m][b]) for b in idx])
                <= CHANCE[key])
        for m in df_ci.index]
df_ci["auc_p"] = [
    np.nan if scores[m] is None else
    np.mean(np.array([roc_auc_score(y_test[b], scores[m][b]) for b in idx]) <= 0.5)
    for m in df_ci.index]

print(f"n_test = {len(y_test)} ({int(y_test.sum())} positiv), "
      f"{N_BOOT} Ziehungen, {dropped} verworfen")
print(f"Zufallsniveau: AUC 0.500 | bal_acc 0.500 | macro-F1 {CHANCE['f1']:.3f}\n")
print(df_ci[COLS + ["auc_p", "bal_acc_p", "f1_p"]].to_string())

# ---------------------------------------------------------------- Export
out = df_ci[COLS + ["auc_p", "bal_acc_p", "f1_p"]].copy()
out["n_test"] = len(y_test)
out["n_positive"] = int(y_test.sum())
out["n_bootstrap"] = N_BOOT
out["chance_auc"] = CHANCE["auc"]
out["chance_bal_acc"] = CHANCE["bal_acc"]
out["chance_f1"] = round(CHANCE["f1"], 3)

#out.to_csv("model_performance_bootstrap.csv")                                       #saving of performance


#--------------------------------------------------------------------------------------------------------------------------feature importance0

feature_importance = {}

for name, pipe in fitted_pipelines.items():
    perm_result = permutation_importance(
        pipe, 
        X_test, 
        y_test, 
        scoring="balanced_accuracy", 
        n_repeats=30, 
        random_state=42,
        n_jobs=-1
    )

    perm_imp = pd.Series(perm_result.importances_mean, index=feature_names).sort_values(ascending=False)

    feature_importance[name] = perm_imp
    
    print(f"\n--- {name} (Top 5 Permutation Importances) ---")
    print(perm_imp.head(5))

df_importance = pd.DataFrame(feature_importance)

#df_importance.to_csv(r"feature_importance.csv", sep = ",")






