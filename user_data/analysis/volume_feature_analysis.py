import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CSV einlesen ---
df = pd.read_csv("user_data/data/volume_features_BTC_USDT.csv", index_col=0)

# --- 2. Zielvariablen berechnen: zukünftige Preisänderungen ---
df['future_close_1'] = df['close'].shift(-1)
df['future_close_6'] = df['close'].shift(-6)     # ~30min bei 5m-Candles
df['future_close_12'] = df['close'].shift(-12)   # ~1h

df['target_1'] = (df['future_close_1'] - df['close']) / df['close']
df['target_6'] = (df['future_close_6'] - df['close']) / df['close']
df['target_12'] = (df['future_close_12'] - df['close']) / df['close']

# --- 3. Korrelationen berechnen ---
features = ['volume', 'obv', 'cvd']
targets = ['target_1', 'target_6', 'target_12']

print("\nKorrelationen der Volumen-Features mit zukünftigen Preisänderungen:")
corr = df[features + targets].corr()
print(corr.loc[features, targets])

# --- 4. Feature-Signale plotten und untersuchen ---
plt.figure(figsize=(15, 8))
ax1 = plt.subplot(311)
ax1.plot(df['close'], label='Price', color='black')
ax1.set_ylabel("Price")
ax1.legend()
ax2 = plt.subplot(312, sharex=ax1)
ax2.plot(df['cvd'], label='CVD', color='blue')
ax2.set_ylabel("CVD")
ax2.legend()
ax3 = plt.subplot(313, sharex=ax1)
ax3.plot(df['obv'], label='OBV', color='green')
ax3.set_ylabel("OBV")
ax3.legend()
plt.suptitle("BTC/USDT: Preis, CVD, OBV")
plt.show()

# --- 5. Signal-Engineering: Beispiele für Volumen-Crosses ---
# (Wo ändert sich cvd/obv stark? Folgt der Kurs?)

df['cvd_delta'] = df['cvd'].diff()
df['obv_delta'] = df['obv'].diff()

# Schwellenwerte testen: Wenn cvd_delta oder obv_delta besonders hoch, wie verhält sich target_6?
threshold_cvd = df['cvd_delta'].std() * 2
threshold_obv = df['obv_delta'].std() * 2

signal_cvd = df[abs(df['cvd_delta']) > threshold_cvd]
signal_obv = df[abs(df['obv_delta']) > threshold_obv]

print("\nWenn CVD starken Schub bekommt (>|%.2f|):" % threshold_cvd)
print(signal_cvd['target_6'].describe())

print("\nWenn OBV starken Schub bekommt (>|%.2f|):" % threshold_obv)
print(signal_obv['target_6'].describe())

# Visualisierung: Histogramm der target_6 bei CVD/OBV-Schüben
plt.figure(figsize=(12,6))
sns.histplot(signal_cvd['target_6'], bins=40, color='blue', label='CVD Spike', kde=True)
sns.histplot(signal_obv['target_6'], bins=40, color='green', label='OBV Spike', kde=True)
plt.axvline(0, color='black', linestyle='--')
plt.legend()
plt.title("Future Returns bei Volumen-Spikes")
plt.xlabel("6x5min-Future-Return (ca. 30min)")
plt.show()
