from freqtrade.strategy import IStrategy
import ta
import pandas as pd
import numpy as np

class VolumeFeatureStrategy(IStrategy):
    """
    Feature-Engineering-Strategie auf Basis von Volume, OBV und CVD.
    Exportiert alle relevanten Features als CSV für weitere Analyse.
    """
    timeframe = '5m'
    minimal_roi = {"0": 0.02}
    stoploss = -0.03
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Volume direkt aus Daten
        dataframe['volume'] = dataframe['volume']

        # On-Balance Volume (ta)
        dataframe['obv'] = ta.volume.on_balance_volume(close=dataframe['close'], volume=dataframe['volume'])

        # Cumulative Volume Delta (CVD) - OHLCV-Approximierung
        # "Buy Volumen" = Volumen wenn close > open, "Sell Volumen" = Volumen wenn close < open
        dataframe['buy_vol'] = np.where(dataframe['close'] > dataframe['open'], dataframe['volume'], 0)
        dataframe['sell_vol'] = np.where(dataframe['close'] < dataframe['open'], dataframe['volume'], 0)
        dataframe['delta_vol'] = dataframe['buy_vol'] - dataframe['sell_vol']
        dataframe['cvd'] = dataframe['delta_vol'].cumsum()

        # Export für Analyse (letzte 2000 Kerzen)
        if len(dataframe) > 0:
            dataframe.tail(2000).to_csv(f"user_data/data/volume_features_{metadata['pair'].replace('/', '_')}.csv")

        return dataframe

    # Dummy-Signale (noch kein echtes Trading, nur Analyse)
    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['buy'] = 0
        return dataframe

    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['sell'] = 0
        return dataframe
