import json

code = """# [Section 3-2] 하이퍼파라미터 튜닝 (LR/DT: GridSearch, 나머지: RandomSearch)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint, uniform

# ── 1) 공통 설정 ──────────────────────────────────────────────────
# cv=3: 빠른 탐색을 위해 3-Fold, scoring=F1(불균형 기준)
CV, SCORING, JOBS = 3, "f1", -1

# ── 2) Grid Search: LR, DT (파라미터 공간이 작음) ────────────────
grid_configs = {
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "params": {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"],
        },
    },
    "DecisionTree": {
        "model": DecisionTreeClassifier(class_weight="balanced", random_state=42),
        "params": {
            "max_depth": [4, 6, 8, 10],
            "min_samples_leaf": [1, 5, 10],
        },
    },
}

grid_results = {}
for name, cfg in grid_configs.items():
    gs = GridSearchCV(cfg["model"], cfg["params"], cv=CV, scoring=SCORING, n_jobs=JOBS)
    gs.fit(X_train, Y_train)
    grid_results[name] = gs.best_estimator_
    print(f"[Grid] {name} best params: {gs.best_params_}")

# ── 3) Random Search: RF, GBM, XGBoost, LightGBM, CatBoost ──────
# n_iter=30: 각 모델당 30회 샘플링 (속도·탐색 균형)
ITER = 30

random_configs = {
    "RandomForest": {
        "model": RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=JOBS),
        "params": {
            "n_estimators": randint(100, 400),
            "max_depth": randint(4, 15),
            "min_samples_leaf": randint(1, 10),
            "max_features": ["sqrt", "log2"],
        },
    },
    "GBM": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "n_estimators": randint(100, 400),
            "learning_rate": uniform(0.01, 0.2),
            "max_depth": randint(3, 8),
            "min_samples_leaf": randint(1, 10),
        },
    },
    "XGBoost": {
        "model": XGBClassifier(scale_pos_weight=neg_pos_ratio, eval_metric="logloss",
                               random_state=42, n_jobs=JOBS),
        "params": {
            "n_estimators": randint(100, 400),
            "learning_rate": uniform(0.01, 0.2),
            "max_depth": randint(3, 8),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
        },
    },
    "LightGBM": {
        "model": LGBMClassifier(class_weight="balanced", random_state=42, n_jobs=JOBS, verbose=-1),
        "params": {
            "n_estimators": randint(100, 400),
            "learning_rate": uniform(0.01, 0.2),
            "num_leaves": randint(20, 80),
            "min_child_samples": randint(10, 50),
        },
    },
    "CatBoost": {
        "model": CatBoostClassifier(auto_class_weights="Balanced", random_seed=42, verbose=0),
        "params": {
            "iterations": randint(100, 400),
            "learning_rate": uniform(0.01, 0.2),
            "depth": randint(4, 10),
            "l2_leaf_reg": uniform(1, 9),
        },
    },
}

random_results = {}
for name, cfg in random_configs.items():
    rs = RandomizedSearchCV(cfg["model"], cfg["params"], n_iter=ITER,
                            cv=CV, scoring=SCORING, n_jobs=JOBS, random_state=42)
    rs.fit(X_train, Y_train)
    random_results[name] = rs.best_estimator_
    print(f"[Random] {name} best params: {rs.best_params_}")

# ── 4) 튜닝 전/후 성능 비교표 ──────────────────────────────────────
# 튜닝 전 결과(result_df)와 튜닝 후를 나란히 비교
def get_metrics(model, label):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    return {
        "Model": label,
        "Accuracy" : round(accuracy_score(Y_test, pred), 4),
        "Precision": round(precision_score(Y_test, pred, zero_division=0), 4),
        "Recall"   : round(recall_score(Y_test, pred, zero_division=0), 4),
        "F1"       : round(f1_score(Y_test, pred, zero_division=0), 4),
        "ROC-AUC"  : round(roc_auc_score(Y_test, prob), 4),
    }

tuned_rows = []
for name, model in {**grid_results, **random_results}.items():
    tuned_rows.append(get_metrics(model, name))

tuned_df = pd.DataFrame(tuned_rows).sort_values("F1", ascending=False).reset_index(drop=True)
print("\\n=== 튜닝 후 성능 비교 (F1 기준 정렬) ===")
display(tuned_df)

# 최고 F1 모델 저장 (이후 섹션에서 재사용)
best_model_name = tuned_df.iloc[0]["Model"]
best_model = {**grid_results, **random_results}[best_model_name]
print(f"\\n최고 성능 모델: {best_model_name}")
"""

p = r'c:\Users\Admin\Desktop\AI Autonomous\cursor\13일차\머신러닝 Day 6. 미니프로젝트.ipynb'
nb = json.load(open(p, encoding='utf-8'))
c = next(x for x in nb['cells'] if x.get('id') == 'f0d3a26c')
c['source'] = [line + '\n' for line in code.splitlines()]
c['source'][-1] = c['source'][-1].rstrip('\n')
with open(p, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('saved')
