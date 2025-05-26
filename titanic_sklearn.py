# titanic_sklearn.py  (~35 LOC)
# Scikit-learn
import pandas as pd, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1) --------- Load ---------
df = sns.load_dataset("titanic").dropna(subset=["age", "fare"])
X = df[["pclass", "sex", "age", "fare", "sibsp", "parch", "alone"]]
y = df["survived"]

# 2) --------- Pipeline ---------
num_cols = ["age", "fare", "sibsp", "parch"]
cat_cols = ["pclass", "sex", "alone"]
pre = ColumnTransformer(
    [("num", StandardScaler(), num_cols),
     ("cat", OneHotEncoder(drop="first"), cat_cols)],
)

pipe = Pipeline([
    ("pre", pre),
    ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
])

# 3) --------- Train & evaluate ---------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
pipe.fit(X_train, y_train)
print("Scikit-learn accuracy:", accuracy_score(y_test, pipe.predict(X_test)).round(3))
