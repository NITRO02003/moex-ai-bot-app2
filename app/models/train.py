# app/train.py
import os, glob, pandas as pd, joblib
from app.features import make_features
from app.models.catboost_model import CatModel

def load_merged(parquet_glob):
    dfs = []
    for p in glob.glob(parquet_glob):
        df = pd.read_parquet(p)
        df["ticker"] = os.path.basename(p).split("_")[0]
        dfs.append(df)
    return pd.concat(dfs).sort_values(["ticker","time"])

def prepare_xy(df):
    # таргет: знак доходности на горизонте 3 бара
    df["ret_fwd"] = df.groupby("ticker")["close"].pct_change(-3) * -1  # смещение вперёд
    df["y"] = (df["ret_fwd"] > 0).astype(int)
    feats = make_features(df.rename(columns=str.upper))
    # выравниваем с y
    m = df.set_index("time").loc[feats.index]
    X = feats.drop(columns=[c for c in feats.columns if c.lower() in {"close","open","high","low","volume"}], errors="ignore")
    y = (m["ret_fwd"] > 0).astype(int)
    X = X.loc[y.index].dropna()
    y = y.loc[X.index]
    return X, y

def main():
    os.makedirs("./models", exist_ok=True)
    raw = load_merged("./data/raw/*_1m.parquet")
    X, y = prepare_xy(raw)
    model = CatModel()
    model.fit(X, y)
    joblib.dump(model, "./models/cat_model.joblib")
    print("Model saved to ./models/cat_model.joblib")

if __name__ == "__main__":
    main()
