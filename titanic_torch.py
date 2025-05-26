# titanic_torch.py  (~40 LOC)
# PyTorch
import numpy as np, pandas as pd, seaborn as sns, torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

# 1) --------- Load & preprocess (same as TF) ---------
df = sns.load_dataset("titanic").dropna(subset=["age", "fare"])
X = df[["pclass", "sex", "age", "fare", "sibsp", "parch", "alone"]]
y = df["survived"].values.astype("float32")

num_cols = ["age", "fare", "sibsp", "parch"]
cat_cols = ["pclass", "sex", "alone"]
pre = ColumnTransformer(
    [("num", StandardScaler(), num_cols),
     ("cat", "passthrough", cat_cols)],
    remainder="drop",
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
X_train = torch.tensor(pre.fit_transform(X_train), dtype=torch.float32, device=device)
X_test  = torch.tensor(pre.transform(X_test),  dtype=torch.float32, device=device)
y_train = torch.tensor(y_train, device=device).unsqueeze(1)
y_test  = torch.tensor(y_test , device=device).unsqueeze(1)

# 2) --------- Model, loss, optimiser ---------
model = nn.Sequential(
    nn.Linear(X_train.shape[1], 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Sigmoid(),
).to(device)
loss_fn = nn.BCELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# 3) --------- Train ---------
for epoch in range(20):
    y_pred = model(X_train)
    loss   = loss_fn(y_pred, y_train)
    opt.zero_grad(); loss.backward(); opt.step()

# 4) --------- Evaluate ---------
with torch.no_grad():
    pred = (model(X_test) > .5).cpu().numpy().astype(int).ravel()
print("PyTorch accuracy:", accuracy_score(y_test.cpu(), pred).round(3))
