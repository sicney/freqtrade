# user_data/strategies/TrendVolatilityStrategy.py

from freqtrade.strategy import IStrategy
import ta  # Pure Python-Package, keine Systemlibs nötig

class TrendVolatilityStrategy(IStrategy):
    """
    Trendfolge-Strategie: EMA50/EMA200 Cross, Einstieg nur bei überdurchschnittlicher Volatilität (ATR).
    Kompatibel mit purem Python-Package 'ta'.
    """
    timeframe = '5m'
    minimal_roi = {
        "0": 0.02  # 2% Take Profit
    }
    stoploss = -0.03   # 3% Stoploss
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    def populate_indicators(self, dataframe, metadata):
        # EMA Indikatoren
        dataframe['ema50'] = ta.trend.ema_indicator(close=dataframe['close'], window=50)
        dataframe['ema200'] = ta.trend.ema_indicator(close=dataframe['close'], window=200)
        # ATR + ATR-Mittelwert
        dataframe['atr'] = ta.volatility.average_true_range(
            high=dataframe['high'], low=dataframe['low'], close=dataframe['close'], window=14)
        dataframe['atr_mean'] = dataframe['atr'].rolling(window=100).mean()
        return dataframe

    def populate_buy_trend(self, dataframe, metadata):
        dataframe.loc[
            (
                (dataframe['ema50'] > dataframe['ema200']) &
                (dataframe['ema50'].shift(1) <= dataframe['ema200'].shift(1)) &
                (dataframe['atr'] > dataframe['atr_mean'])
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe, metadata):
        dataframe.loc[
            (
                (dataframe['ema50'] < dataframe['ema200']) |
                (dataframe['close'] > dataframe['close'].shift(1) * 1.02)
            ),
            'sell'] = 1
        return dataframe
