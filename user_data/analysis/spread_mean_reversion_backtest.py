import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==== SETTINGS =====
PAIR1 = "ETHUSDT"
PAIR2 = "ETHDAI"
TIMEFRAME = "15m"
FEE_RATE = 0.001  # 0.1% pro Trade, Roundtrip = 0.2%

DATA_DIR = "user_data/data/binance"
CSV1 = f"{DATA_DIR}/{PAIR1}_{TIMEFRAME}.csv"
CSV2 = f"{DATA_DIR}/{PAIR2}_{TIMEFRAME}.csv"

# ==== LOAD DATA =====
def load_price(csv):
    df = pd.read_csv(csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df["close"].astype(float)

s1 = load_price(CSV1)
s2 = load_price(CSV2)
df = pd.DataFrame({PAIR1: s1, PAIR2: s2}).dropna()

# ==== SPREAD-BERECHNUNG (Lineares Hedge-Ratio) =====
# Hedge-Ratio via linearer Regression, damit Spread stationär ist
import statsmodels.api as sm
X = sm.add_constant(df[PAIR2])
model = sm.OLS(df[PAIR1], X).fit()
hedge_ratio = model.params[PAIR2]

df["spread"] = df[PAIR1] - hedge_ratio * df[PAIR2]
df["spread_mean"] = df["spread"].rolling(100).mean()
df["spread_std"] = df["spread"].rolling(100).std()
df["zscore"] = (df["spread"] - df["spread_mean"]) / df["spread_std"]

# ==== VISUALISIERUNG ====
plt.figure(figsize=(16, 8))
plt.plot(df.index, df["spread"], label="Spread")
plt.plot(df.index, df["spread_mean"], label="Rolling Mean (100)")
plt.fill_between(df.index, df["spread_mean"] + 2 * df["spread_std"],
                 df["spread_mean"] - 2 * df["spread_std"], color="grey", alpha=0.2, label="±2 Std")
plt.title(f"Spread und Rolling Mean - {PAIR1} vs {PAIR2} ({TIMEFRAME})")
plt.legend()
plt.tight_layout()
plt.savefig(f"user_data/analysis/{PAIR1}_{PAIR2}_{TIMEFRAME}_spread.png")
plt.close()

plt.figure(figsize=(16, 4))
plt.plot(df.index, df["zscore"], label="Z-Score")
plt.axhline(2, color="red", linestyle="--")
plt.axhline(-2, color="red", linestyle="--")
plt.axhline(0, color="black", linestyle=":")
plt.title(f"Z-Score des Spreads - {PAIR1} vs {PAIR2} ({TIMEFRAME})")
plt.legend()
plt.tight_layout()
plt.savefig(f"user_data/analysis/{PAIR1}_{PAIR2}_{TIMEFRAME}_zscore.png")
plt.close()

# ==== ADFULLER (STATIONARITÄT) ====
from statsmodels.tsa.stattools import adfuller
adf_result = adfuller(df["spread"].dropna())
with open(f"user_data/analysis/{PAIR1}_{PAIR2}_{TIMEFRAME}_adf.txt", "w") as f:
    f.write(f"ADF Statistic: {adf_result[0]}\n")
    f.write(f"p-value: {adf_result[1]}\n")

# ==== SIMPLE MEAN REVERSION BACKTEST ====
capital = 10000  # USDT Startkapital
position = 0     # +1 = Long Spread, -1 = Short Spread
entry_price = 0
pnl = []
trades = []
fee_paid = 0
in_trade = False

for i in range(1, len(df)):
    # Entry-Logik
    if not in_trade:
        if df["zscore"].iloc[i-1] > 2:
            position = -1  # Short Spread
            entry_price = df["spread"].iloc[i]
            entry_idx = i
            in_trade = True
            fee_paid += capital * FEE_RATE
        elif df["zscore"].iloc[i-1] < -2:
            position = 1  # Long Spread
            entry_price = df["spread"].iloc[i]
            entry_idx = i
            in_trade = True
            fee_paid += capital * FEE_RATE
    # Exit-Logik
    elif in_trade:
        if (position == 1 and abs(df["zscore"].iloc[i-1]) < 0.5) or \
           (position == -1 and abs(df["zscore"].iloc[i-1]) < 0.5):
            # Position schließen
            profit = (df["spread"].iloc[i] - entry_price) * position
            fee_paid += capital * FEE_RATE
            trades.append(profit)
            pnl.append(np.sum(trades) - fee_paid)
            in_trade = False
            position = 0

# ===== REPORT =====
if len(trades) == 0:
    print("Keine Trades ausgeführt!")
    exit()

final_pnl = np.sum(trades) - fee_paid
hit_ratio = np.mean(np.array(trades) > 0)
max_drawdown = np.max(np.maximum.accumulate(np.cumsum(trades)) - np.cumsum(trades))

report = (
    f"\n===== SUMMARY: {PAIR1} / {PAIR2} ({TIMEFRAME}) =====\n"
    f"Anzahl Trades: {len(trades)}\n"
    f"P&L gesamt: {final_pnl:.2f} USDT\n"
    f"Trefferquote: {hit_ratio*100:.2f}%\n"
    f"Max Drawdown: {max_drawdown:.2f} USDT\n"
    f"Total Fees: {fee_paid:.2f} USDT\n"
    f"ADF-Stat: {adf_result[0]:.3f}, p-value: {adf_result[1]:.3e}\n"
)

print(report)

# Trade-by-Trade-Report:
print("\n==== Einzelne Trades (Profit pro Trade) ====")
for i, trade in enumerate(trades):
    print(f"Trade {i+1}: {'Gewinn' if trade>0 else 'Verlust'}: {trade:.2f} USDT")

# Plot sofort anzeigen (optional, kann Terminal blockieren):
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.plot(np.cumsum(trades) - fee_paid, label="Cumulative P&L")
plt.title(f"Cumulative P&L ({PAIR1}/{PAIR2}, {TIMEFRAME})")
plt.xlabel("Trade")
plt.ylabel("Profit (USDT)")
plt.legend()
plt.tight_layout()
plt.show()  # Zeigt Plot direkt im Codespace an, wenn möglich


