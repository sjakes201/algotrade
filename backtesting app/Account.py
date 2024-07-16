import numpy as np
from Position import Position

class Account:

    def __init__(self, name, starting_balance, algo_params):
        self.positions = []
        self.per_contract_fee = 0
        self.name = name
        self.balance = starting_balance
        self._starting_balance = starting_balance
        self.total_cost_from_fees = 0
        self.algo_params = algo_params
        self._next_position_id = 1
        self.low = self.high = starting_balance
        self.today_trades = 0
        self.profit_trades = 0
        self.losing_trades = 0
        self.daily_trades = []
        self.today_profit = 0
        self.daily_profits = []

    def __str__(self):
        profit = self.balance - self._starting_balance
        winrate = self.profit_trades / (self.profit_trades + self.losing_trades) * 100
        acc_summary = (
            "***** \n"
            f"Account {self.name} \n"
            f"Starting Balance: ${self._starting_balance:.2f} \n"
            f"Balance: ${self.balance:.2f} \n"
            f"Realized Profit: ${profit:.2f} \n"
            f"Total Fees: ${self.total_cost_from_fees:.2f} \n"
            f"Balance High: ${self.high:.2f} \n"
            f"Balance Low: ${self.low:.2f} \n"
            f"Total positions: {len(self.positions)} \n"
            f"Daily trades: {self.daily_trades} \n"
            f"Daily profits: {self.daily_profits} \n"
            f"Win rate: {winrate:.2f}% \n"
            f"Algo params: {self.algo_params} \n"
            "***** \n"
        )

        return acc_summary        

    def set_model(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor

    def set_contract_config(self, per_contract_fee = 0, point_dollar_value = 5, points_slippage = 0):
        self.per_contract_fee = per_contract_fee
        self.point_dollar_value = point_dollar_value
        self.points_slippage = points_slippage

    """
        Symbol (string) is the future contract name
        Quantity (int) is the number of contracts
        Direction (string) is "LONG' or "SHORT"
        Price (float) is the current price of the contract
        Time (datetime) is the time the order was placed
        
    """
    def order_to_open(self, symbol, direction, recent_price, time):
        self.total_cost_from_fees -= self.per_contract_fee
        self.balance -= self.per_contract_fee
        take_profit_diff = self.algo_params['take_profit_diff']
        stop_loss_diff = self.algo_params['stop_loss_diff']
        position_id = self._next_position_id
        self._next_position_id += 1
        position = Position(position_id, symbol, direction, recent_price, time, stop_loss_diff, take_profit_diff, trailing_loss=self.algo_params['trailing_loss'])
        self.positions.append(position)
        self.today_trades += 1
        
    
    
    def update_balance(self, amount):
        self.balance += amount
        if self.balance < self.low:
            self.low = self.balance
        if self.balance > self.high:
            self.high = self.balance
    
    def get_balance(self):
        return self.balance
    
    def get_total_fees(self):
        return self.total_cost_from_fees
    
    """
        Symbol (string) is the future contract name
        Quantity (int) is the number of contracts you want to exit
        Direction (string) is "LONG' or "SHORT" for what you want to exit, or decrease the position in
        Price (float) is the current price of the contract
    """
    def order_to_close(self, symbol, direction, recent_price, position_id=None):
        # Search for a position to close
        for i, d in enumerate(self.positions):
            if position_id is None:
                # If position_id is None, use symbol and direction to find the position
                if d.symbol == symbol and d.type == direction:
                    self._close_position(i, d, recent_price)
                    break
            else:
                # If position_id is provided, use it to find the specific position
                if d.get_id() == position_id:
                    self._close_position(i, d, recent_price)
                    break

    def _close_position(self, index, position, recent_price):
        removed = self.positions.pop(index)
        max_profit = removed.get_max_profit()
        max_loss = removed.get_max_loss()
        profit = removed.get_unrealized_profit(recent_price)
        # Assume max loss and max profit would've hit (with slippage)

        if profit > max_profit:
            profit = max_profit
        elif profit < max_loss:
            profit = max_loss

        self.total_cost_from_fees -= self.per_contract_fee
        self.balance -= self.per_contract_fee

        # Calculate profit/loss
        self.update_balance(profit)
        self.today_profit += profit
        self.today_trades += 1
        if profit > 5:
            self.profit_trades += 1
        elif profit < -5:
            self.losing_trades += 1


    def get_positions(self):
        return self.positions
    
    """
        Requires df with all calc_data columns
        Will return string 'BUY', 'SELL', or 'HOLD' based on the most recent minutes prediction
    """
    def get_recent_prediction(self, df):
        feature_columns = ['momentum_signal', 'mean_reversion_signal', 'breakout_signal',
       'reversal_signal', 'vwap_trend_signal', 'ema_12_26_signal',
       'scalping_signal', 'heikin_ashi_signal', 'market_regime_ADX',
       'market_regime_MA', 'volatility', 'total_buys', 'total_sells',
       'momentum_signal_buy_accum', 'momentum_signal_sell_accum',
       'mean_reversion_signal_buy_accum', 'mean_reversion_signal_sell_accum',
       'breakout_signal_buy_accum', 'breakout_signal_sell_accum',
       'reversal_signal_buy_accum', 'reversal_signal_sell_accum',
       'vwap_trend_signal_buy_accum', 'vwap_trend_signal_sell_accum',
       'ema_12_26_signal_buy_accum', 'ema_12_26_signal_sell_accum',
       'scalping_signal_buy_accum', 'scalping_signal_sell_accum',
       'heikin_ashi_signal_buy_accum', 'heikin_ashi_signal_sell_accum']

        def determine_label(probabilities, confidence_threshold):
            # Apply the threshold to get predictions
            if max(probabilities) < confidence_threshold:
                return 'HOLD'
            else:
                predicted_class = np.argmax(probabilities)
                if predicted_class == 2:
                    return 'BUY'
                elif predicted_class == 0:
                    return 'SELL'
                else:
                    return 'HOLD'

        prepared_df = df.loc[:, feature_columns]
        X_processed = self.preprocessor.transform(prepared_df)
        probabilities = self.model.predict_proba(X_processed)
        return determine_label(probabilities[0], self.algo_params['confidence_threshold'])
        
    def get_net_position(self, symbol):
        # Parse through positions and find net quantity of positions. Positive is long, negative is short
        net = 0
        for position in self.positions:
            if position.symbol == symbol:
                if position.type == 'LONG':
                    net += 1
                elif position.type == 'SHORT':
                    net -= 1
        return net
        
    def check_all_positions(self, symbol, close, high, low):
        # Filter the positions first
        filtered_positions = [p for p in self.positions if p.symbol == symbol]
        
        # Iterate over the filtered positions
        for position in filtered_positions:
            position.update_trailing_stop(close)
            result = position.check_action_from_price(close, high, low)
            if result in ('stop_loss', 'take_profit'):
                position_id = position.get_id()
                self.order_to_close(position.symbol, position.type, close, position_id=position_id)

    def close_all_positions(self, symbol, new_price):
        # Filter the positions first
        filtered_positions = [p for p in self.positions if p.symbol == symbol]
        
        # Iterate over the filtered positions
        for position in filtered_positions:
            position_id = position.get_id()
            self.order_to_close(position.symbol, position.type, new_price, position_id=position_id)
        
    def process_recent_candle(self, df_row):
        symbol = df_row['symbol'].values[0]
        close = df_row['close'].values[0]
        high = df_row['high'].values[0]
        low = df_row['low'].values[0]
        self.check_all_positions(symbol, close, high, low)
        
        trade_order = self.get_recent_prediction(df_row)
        net_pos = self.get_net_position(df_row['symbol'].values[0])
        
        if trade_order == "BUY":
            if net_pos >= self.algo_params['max_position_size']:
                # Too large a long position already
                return
            elif net_pos >= 0:
                # Submit one long order
                self.order_to_open(df_row['symbol'].values[0], "LONG", df_row['close'].values[0], df_row['datetime_est'].values[0])
            elif net_pos == -1:
                # exit one short and enter one long
                self.order_to_close(df_row['symbol'].values[0], "SHORT", df_row['close'].values[0])
                self.order_to_open(df_row['symbol'].values[0], "LONG", df_row['close'].values[0], df_row['datetime_est'].values[0])
            else:
                # exit two short
                self.order_to_close(df_row['symbol'].values[0], "SHORT", df_row['close'].values[0])
                self.order_to_close(df_row['symbol'].values[0], "SHORT", df_row['close'].values[0])
        elif trade_order == "SELL":
            if net_pos <= -1 * self.algo_params['max_position_size']:
                # Too large a short position already
                return
            elif net_pos <= 0:
                # Submit one short order
                self.order_to_open(df_row['symbol'].values[0], "SHORT", df_row['close'].values[0], df_row['datetime_est'].values[0])
            elif net_pos == 1:
                # exit one long and enter one short
                self.order_to_close(df_row['symbol'].values[0], "LONG", df_row['close'].values[0])
                self.order_to_open(df_row['symbol'].values[0], "SHORT", df_row['close'].values[0], df_row['datetime_est'].values[0])
            else:
                # exit two long positions
                self.order_to_close(df_row['symbol'].values[0], "LONG", df_row['close'].values[0])
                self.order_to_close(df_row['symbol'].values[0], "LONG", df_row['close'].values[0])
    
    def log_day(self):
        self.daily_trades.append(self.today_trades)
        self.today_trades = 0
        self.daily_profits.append(self.today_profit)
        self.today_profit = 0