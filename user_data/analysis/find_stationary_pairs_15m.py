import ccxt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm
import time

# --- Einstellungen ---
QUOTE = 'USDT'
TIMEFRAME_H = '1h'
TIMEFRAME_15M = '15m'
LIMIT_H = 500
LIMIT_15M = 500

exchange = ccxt.binance({
    'rateLimit': 1200,
    'enableRateLimit': True
})

print("Lade alle Märkte von Binance...")
markets = exchange.load_markets()
pairs = [s for s in markets if s.endswith('/' + QUOTE) and not s.startswith('BUSD/')]

# Schritt 1: 1h-Scan aller Paare
ohlcvs_h = {}
for symbol in tqdm(pairs, desc="Lade 1h OHLCV-Daten"):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME_H, limit=LIMIT_H)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        ohlcvs_h[symbol] = df['close']
        time.sleep(0.07)
    except Exception as e:
        print(f"Fehler beim Laden von {symbol}: {e}")

all_df_h = pd.concat(ohlcvs_h.values(), axis=1, join='inner')
all_df_h.columns = list(ohlcvs_h.keys())
print(f"{all_df_h.shape[1]} Paare mit vollständigen 1h-Daten.")

# Cointegration-Test auf 1h
results_h = []
symbols = all_df_h.columns.tolist()
for i in tqdm(range(len(symbols)), desc="Vergleiche 1h-Paare"):
    for j in range(i+1, len(symbols)):
        s1 = all_df_h[symbols[i]]
        s2 = all_df_h[symbols[j]]
        beta = np.polyfit(s2, s1, 1)[0]
        spread = s1 - beta * s2
        try:
            adf_result = adfuller(spread)
            pvalue = adf_result[1]
            stat = adf_result[0]
            results_h.append({
                'pair1': symbols[i],
                'pair2': symbols[j],
                'beta': beta,
                'adf_stat_1h': stat,
                'pvalue_1h': pvalue
            })
        except Exception:
            continue

df_h = pd.DataFrame(results_h)
df_h = df_h.sort_values('pvalue_1h')
df_h = df_h[df_h['pvalue_1h'] < 0.1]
print(f"Top {len(df_h)} stationäre Paare auf 1h.")

# Schritt 2: Wähle die besten 50 Paare
df_h_top50 = df_h.head(50)
print(df_h_top50[['pair1','pair2','pvalue_1h']])

# Schritt 3: Ziehe 15m-Daten NUR für die Top-Paare
all_needed = list(set(df_h_top50['pair1']).union(set(df_h_top50['pair2'])))
ohlcvs_15m = {}
for symbol in tqdm(all_needed, desc="Lade 15m OHLCV-Daten (nur Top-Paare)"):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME_15M, limit=LIMIT_15M)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        ohlcvs_15m[symbol] = df['close']
        time.sleep(0.07)
    except Exception as e:
        print(f"Fehler beim Laden von {symbol}: {e}")

all_df_15m = pd.concat(ohlcvs_15m.values(), axis=1, join='inner')
all_df_15m.columns = list(ohlcvs_15m.keys())
print(f"{all_df_15m.shape[1]} Paare mit 15m-Daten.")

# Schritt 4: Cointegration auf 15m für die 50 besten Paare
results_15m = []
for _, row in tqdm(df_h_top50.iterrows(), total=len(df_h_top50), desc="Teste 15m-Stationarität"):
    p1, p2 = row['pair1'], row['pair2']
    try:
        s1 = all_df_15m[p1]
        s2 = all_df_15m[p2]
        beta = np.polyfit(s2, s1, 1)[0]
        spread = s1 - beta * s2
        adf_result = adfuller(spread)
        pvalue = adf_result[1]
        stat = adf_result[0]
        results_15m.append({
            'pair1': p1,
            'pair2': p2,
            'beta_15m': beta,
            'adf_stat_15m': stat,
            'pvalue_15m': pvalue,
            'pvalue_1h': row['pvalue_1h']
        })
    except Exception:
        continue

df_15m = pd.DataFrame(results_15m)
df_15m = df_15m[df_15m['pvalue_15m'] < 0.05].sort_values('pvalue_15m')

print("Paare mit nachgewiesener Stationarität auf 15m:")
print(df_15m[['pair1','pair2','beta_15m','pvalue_1h','pvalue_15m']])

df_15m.to_csv("stationary_pairs_15m.csv", index=False)
