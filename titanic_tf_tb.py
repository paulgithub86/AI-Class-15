# titanic_tf_tb.py
import numpy as np, pandas as pd, seaborn as sns, tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from datetime import datetime
import os

# ---------- Data prep (identical to previous script) ----------
df = sns.load_dataset("titanic").dropna(subset=["age", "fare"])
X = df[["pclass", "sex", "age", "fare", "sibsp", "parch", "alone"]]
y = df["survived"].values.astype("float32")

num_cols = ["age", "fare", "sibsp", "parch"]
cat_cols = ["pclass", "sex", "alone"]
pre = ColumnTransformer(
    [("num", StandardScaler(), num_cols),
     ("cat", "passthrough", cat_cols)],
    verbose_feature_names_out=False)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, stratify=y, random_state=42)

X_train = pre.fit_transform(X_train).astype("float32")
X_test  = pre.transform(X_test).astype("float32")

# ---------- Model ----------
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(X_train.shape[1]),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1,  activation="sigmoid"),
])
model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

# ---------- TensorBoard callback (4 new lines) ----------
log_dir = os.path.join("logs", "titanic",
                       datetime.now().strftime("%Y%m%d-%H%M%S"))
tb_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                       histogram_freq=1)  # layer‐wise weights

# ---------- Train ----------
model.fit(X_train, y_train,
          epochs=20,
          batch_size=32,
          callbacks=[tb_cb],
          validation_split=0.2,
          verbose=0)

# ---------- Evaluate ----------
pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).ravel()
print("TensorFlow (+TB) accuracy:", accuracy_score(y_test, pred).round(3))
print(f"Logs saved to: {log_dir}")
