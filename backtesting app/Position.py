class Position:

    
    """
        symbol: string for future contract name
        type: string "LONG" or "SHORT",
        entry_price: price of candle when entered
        time_filled: **TO DETERMINE**
        stop_loss_diff: int for number of points away from entry price to set stop loss
        take_profit_diff: int for number of points away from entry price to set take profit
        trailing_loss: bool for whether to use trailing stop loss
    """
    def __init__(self, position_id, symbol, type, entry_price, time_filled, stop_loss_diff = 0, take_profit_diff = 0, trailing_loss = False, point_dollar_value = 5, points_slippage = 0):
        self.position_id = position_id
        self.symbol = symbol
        self.type = type # "LONG" or "SHORT"
        self.time_filled = time_filled
        self.entry_price = entry_price
        self.stop_loss = None # Threshold in points 
        self.take_profit = None # Threshold in points
        self.max_loss = float('-inf') # Amount in dollars
        self.max_profit = float('inf') # Amount in dollars
        self.point_dollar_value = point_dollar_value # Dollars per future contact points
        self.points_slippage = points_slippage # Slippage in points that goes against order filling
        
        # Define max profit and loss if the legs exist
        if take_profit_diff != 0:
            self.max_profit = (take_profit_diff - self.points_slippage) * self.point_dollar_value
        if stop_loss_diff != 0:
            self.max_loss = -1 * (stop_loss_diff + self.points_slippage) * self.point_dollar_value
        
        # Define stop loss and take profit
        if type == "LONG":
            if take_profit_diff != 0:
                self.take_profit = {
                    'price': entry_price + take_profit_diff,
                    'take_profit_diff': take_profit_diff,
                }
            if stop_loss_diff != 0:
                self.stop_loss = {
                    'price': entry_price - stop_loss_diff,
                    'trailing_stop': trailing_loss,
                    'stop_loss_diff': stop_loss_diff,
                }
        elif type == "SHORT":
            if take_profit_diff != 0:
                self.take_profit = {
                    'price': entry_price - take_profit_diff,
                    'take_profit_diff': take_profit_diff,
                }
            if stop_loss_diff != 0:
                self.stop_loss = {
                    'price': entry_price + stop_loss_diff,
                    'trailing_stop': trailing_loss,
                    'stop_loss_diff': stop_loss_diff,
                }
        
    def __str__(self):
        return (f"Position ID: {self.position_id}\n"
                f"Symbol: {self.symbol}\n"
                f"Type: {self.type}\n"
                f"Time Filled: {self.time_filled}\n"
                f"Entry Price: {self.entry_price}\n"
                f"Stop Loss: {self.stop_loss}\n"
                f"Take Profit: {self.take_profit}\n"
                f"Max Loss: ${self.max_loss}\n"
                f"Max Profit: ${self.max_profit}\n")   
    
    def get_unrealized_profit(self, new_price):
        if self.type == 'LONG':
            return (new_price - self.entry_price) * self.point_dollar_value
        elif self.type == 'SHORT':
            return (self.entry_price - new_price) * self.point_dollar_value
        else:
            print(f"ERROR: Invalid type {self.type} for position")
            return 0
        
    def check_action_from_price(self, close, high, low):
        # Check if stop loss or profit take are triggered
        if self.type == 'LONG':
            if self.stop_loss is not None and low <= self.stop_loss['price']:
                return 'stop_loss'
            elif self.take_profit is not None and high >= self.take_profit['price']:
                return 'take_profit'
            else:
                return None
        elif self.type == 'SHORT':
            if self.stop_loss is not None and high >= self.stop_loss['price']:
                return 'stop_loss'
            elif self.take_profit is not None and low <= self.take_profit['price']:
                return 'take_profit'
            else:
                return None
        else:
            print(f"ERROR: Invalid type {self.type} for position")
            return None
        
    def update_trailing_stop(self, new_price):
        if not self.stop_loss.get('trailing_stop', False):
            return
        
        if self.type == 'LONG':
            if new_price - self.stop_loss['price'] > self.stop_loss['stop_loss_diff']:
                self.stop_loss['price'] = new_price - self.stop_loss['stop_loss_diff']
        elif self.type == 'SHORT':
            if self.stop_loss['price'] - new_price > self.stop_loss['stop_loss_diff']:
                self.stop_loss['price'] = new_price + self.stop_loss['stop_loss_diff']
        else:
            raise ValueError(f"ERROR: Invalid type {self.type} for position")


        
    def get_max_loss(self):
        return self.max_loss

    def get_max_profit(self):
        return self.max_profit
    
    def get_id(self):
        return self.position_id