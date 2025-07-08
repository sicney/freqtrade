# user_data/analysis/find_stationary_pairs_local.py

import os
import glob
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm

DATA_DIR = "user_data/data/binance"
TIMEFRAMES = ["5m", "15m", "1h"]
MIN_LEN = 800  # Min. Kerzenanzahl für Paar-Analyse (anpassen je nach Historie)

def load_prices(symbol, timeframe):
    fpath = os.path.join(DATA_DIR, f"{symbol}_{timeframe}.feather")
    if os.path.exists(fpath):
        df = pd.read_feather(fpath)
        df = df.set_index("date")
        return df["close"]
    else:
        return None

def all_symbols(timeframe):
    files = glob.glob(os.path.join(DATA_DIR, f"*_{timeframe}.feather"))
    return [os.path.basename(f).replace(f"_{timeframe}.feather", "") for f in files]

def test_stationarity(s1, s2):
    if len(s1) != len(s2) or len(s1) < MIN_LEN:
        return None
    # Lineare Regression für Hedge-Ratio (Beta)
    beta = np.polyfit(s2, s1, 1)[0]
    spread = s1 - beta * s2
    # ADF-Test auf Stationarität
    result = adfuller(spread)
    pvalue = result[1]
    return pvalue, beta, spread

for timeframe in TIMEFRAMES:
    print(f"\n=== Suche stationäre Paare im {timeframe} ===")
    symbols = all_symbols(timeframe)
    # Typische Filter, z.B. USDT-, ETH-, BTC-Paare
    usdt = [s for s in symbols if s.endswith("USDT")]
    btc  = [s for s in symbols if s.endswith("BTC")]
    eth  = [s for s in symbols if s.endswith("ETH")]
    all_candidates = list(set(usdt + btc + eth))

    results = []
    for i, s1 in enumerate(tqdm(all_candidates, desc=f"Vergleiche Paare ({timeframe})")):
        price1 = load_prices(s1, timeframe)
        if price1 is None: continue
        for s2 in all_candidates[i+1:]:
            if s1 == s2: continue
            price2 = load_prices(s2, timeframe)
            if price2 is None: continue
            # Kürze auf gleiche Länge
            l = min(len(price1), len(price2))
            s1p = price1[-l:].reset_index(drop=True)
            s2p = price2[-l:].reset_index(drop=True)
            out = test_stationarity(s1p, s2p)
            if out is None: continue
            pvalue, beta, spread = out
            if pvalue < 0.05:
                results.append({
                    "pair1": s1,
                    "pair2": s2,
                    "timeframe": timeframe,
                    "pvalue": pvalue,
                    "beta": beta,
                    "n": l
                })

    df = pd.DataFrame(results).sort_values("pvalue")
    fname = f"stationary_pairs_{timeframe}_local.csv"
    if not df.empty:
        df.to_csv(fname, index=False)
        print(f"{len(df)} stationäre Paare gefunden. Ergebnis gespeichert in {fname}.")
        print(df.head(10))
    else:
        print("Keine stationären Paare gefunden.")

