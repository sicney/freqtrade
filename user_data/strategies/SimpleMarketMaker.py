from freqtrade.strategy import IStrategy
import pandas as pd

class SimpleMarketMaker(IStrategy):
    """
    Einfache Market Maker/Grid-Strategie für Krypto: platziert Buy/Sell auf Grid-Levels um den aktuellen Preis.
    Nimmt kleine Gewinne mit, minimiert Drawdown durch Stoploss. Kein echtes Inventory-Management!
    """
    timeframe = '5m'
    minimal_roi = {
        "0": 0.004  # TP bei 0.4%
    }
    stoploss = -0.008    # SL bei -0.8%
    trailing_stop = False

    # Grid Settings
    grid_pct = 0.003  # 0.3% Abstand um den aktuellen Preis
    max_open_trades = 3

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Mittlerer Preis als MA20
        dataframe['midprice'] = dataframe['close'].rolling(window=20).mean()
        return dataframe

    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Buy-Signal: Kurs unterhalb von midprice - grid_pct
        grid_level = dataframe['midprice'] * (1 - self.grid_pct)
        dataframe.loc[
            (dataframe['close'] < grid_level),
            'buy'
        ] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Sell-Signal: Kurs oberhalb von midprice + grid_pct (für Short oder für das Schließen von Longs)
        grid_level = dataframe['midprice'] * (1 + self.grid_pct)
        dataframe.loc[
            (dataframe['close'] > grid_level),
            'sell'
        ] = 1
        return dataframe
