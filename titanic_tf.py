# titanic_tf.py  (~40 LOC)
# TensorFlow (Keras)
import numpy as np, pandas as pd, seaborn as sns, tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# 1) --------------- Load & preprocess ---------------
df = sns.load_dataset("titanic").dropna(subset=["age", "fare"])  # keep it simple
X = df[["pclass", "sex", "age", "fare", "sibsp", "parch", "alone"]]
y = df["survived"].values.astype("float32")

num_cols = ["age", "fare", "sibsp", "parch"]
cat_cols = ["pclass", "sex", "alone"]
pre = ColumnTransformer(
    [("num", StandardScaler(), num_cols),
     ("cat", "passthrough", cat_cols)],
    remainder="drop",
    verbose_feature_names_out=False,
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
X_train = pre.fit_transform(X_train).astype("float32")
X_test  = pre.transform(X_test).astype("float32")

# 2) --------------- Build & train ---------------
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(X_train.shape[1]),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1,  activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

# 3) --------------- Evaluate ---------------
pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).ravel()
print("TensorFlow accuracy:", accuracy_score(y_test, pred).round(3))
