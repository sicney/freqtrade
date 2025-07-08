import os
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm

# === EINSTELLUNGEN ===
DATA_DIR = 'data/binance'
TIMEFRAMES = ['1h', '15m', '5m']
PVAL_THRESHOLD = 0.1
MIN_LENGTH = 100  # Mindestanzahl gemeinsamer Datenpunkte

def load_close(symbol, timeframe):
    fname = f"{DATA_DIR}/{symbol}_{timeframe}.csv"
    if not os.path.exists(fname):
        return None
    df = pd.read_csv(fname)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    if 'close' in df.columns:
        return df['close']
    return None

def find_all_symbols(timeframe):
    files = os.listdir(DATA_DIR)
    symbols = [f.replace(f'_{timeframe}.csv','') for f in files if f.endswith(f'_{timeframe}.csv')]
    return symbols

for timeframe in TIMEFRAMES:
    print(f"\nSuche stationäre Paare für {timeframe} ...")
    symbols = find_all_symbols(timeframe)
    closes = {}
    for s in tqdm(symbols, desc=f"Lade Close-Daten für {timeframe}"):
        close = load_close(s, timeframe)
        if close is not None and len(close) >= MIN_LENGTH:
            closes[s] = close

    pairs = []
    syms = list(closes.keys())
    for i in tqdm(range(len(syms)), desc=f"Vergleiche Paare für {timeframe}"):
        for j in range(i+1, len(syms)):
            s1 = closes[syms[i]]
            s2 = closes[syms[j]]
            df_pair = pd.concat([s1, s2], axis=1, join='inner').dropna()
            if df_pair.shape[0] < MIN_LENGTH:
                continue
            s1a, s2a = df_pair.iloc[:, 0], df_pair.iloc[:, 1]
            try:
                beta = np.polyfit(s2a, s1a, 1)[0]
                spread = s1a - beta * s2a
                adf_result = adfuller(spread)
                pvalue = adf_result[1]
                stat = adf_result[0]
                if pvalue < PVAL_THRESHOLD:
                    pairs.append({
                        'pair1': syms[i],
                        'pair2': syms[j],
                        'beta': beta,
                        'adf_stat': stat,
                        'pvalue': pvalue,
                        'n': len(spread)
                    })
            except Exception:
                continue

    if not pairs:
        print(f"Keine stationären Paare für {timeframe} gefunden.")
    else:
        dfp = pd.DataFrame(pairs).sort_values('pvalue')
        print(f"Gefundene stationäre Paare für {timeframe}:")
        print(dfp[['pair1', 'pair2', 'pvalue', 'n']].head(10))
        outname = f'stationary_pairs_{timeframe}.csv'
        dfp.to_csv(outname, index=False)
        print(f"Ergebnis gespeichert in: {outname}")
