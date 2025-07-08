import ccxt
import pandas as pd
import os
import time
from tqdm import tqdm

# === EINSTELLUNGEN ===
TIMEFRAMES = ['5m', '15m', '1h']
LIMITS = {'5m': 1000, '15m': 1000, '1h': 1000}   # max 1000 Kerzen pro Request
PAUSE = 0.08    # Binance API Limitierung
DATA_DIR = 'data/binance'
os.makedirs(DATA_DIR, exist_ok=True)

exchange = ccxt.binance({
    'enableRateLimit': True,
    'rateLimit': 1200,
})

print("Lade alle aktiven Binance-Paare ...")
markets = exchange.load_markets()
symbols = []
for s in markets:
    if not markets[s]['active']:
        continue
    # xxx/USDT, ETH/xxx, BTC/xxx
    if s.endswith('/USDT') or s.startswith('ETH/') or s.startswith('BTC/'):
        symbols.append(s)

print(f"Gefunden: {len(symbols)} Paare.")
print("Starte Download für 5m, 15m, 1h ...")

def save_ohlcv(symbol, tf):
    fname = f"{DATA_DIR}/{symbol.replace('/', '')}_{tf}.csv"
    if os.path.exists(fname):
        return  # Bereits geladen
    try:
        print(f"Lade {symbol} {tf} ...")
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=LIMITS[tf])
        if not ohlcv:
            print(f"Leere Daten für {symbol} {tf}")
            return
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.to_csv(fname, index=False)
        time.sleep(PAUSE)
    except Exception as e:
        print(f"Fehler bei {symbol} {tf}: {e}")

for tf in TIMEFRAMES:
    for s in tqdm(symbols, desc=f"Lade {tf} Daten"):
        save_ohlcv(s, tf)

print("Fertig!")
