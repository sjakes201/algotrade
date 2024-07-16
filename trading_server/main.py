from connect import connect
from services import tv_get, tv_post
from handle_account_list import handle_account_list
from services import get_account_balance
import time
from dotenv import load_dotenv
import os
load_dotenv()

credentials = {
    # Fill this with your credentials
    'name': os.getenv('NAME'),
    'password': os.getenv('PASSWORD'),
    'appId': 'Population trader',
    'appVersion': '1.0',
    'deviceId': '807',
    'cid': os.getenv('CID'),
    'sec': os.getenv('SECRET')
}

oso_orders_acc_0 = []
oso_orders_acc_1 = []
oso_orders_acc_2 = []
oso_orders_acc_3 = []
live_oso_orders = []

oso_orders_from_acc_id = {
    10464470: oso_orders_acc_0,
    10664702: oso_orders_acc_1,
    10744564: oso_orders_acc_2,
    10796357: oso_orders_acc_3,
    370703: live_oso_orders
}

def get_order_details(id, env='demo'):
    endpoint = f'/order/item'
    order_data = {
        'id': id,
        'isAutomated': True
    }
    return tv_get(endpoint, order_data, env=env)

def get_working_orders(accountId, env='demo'):
    all_orders = tv_get('/order/list', {'isAutomated': True}, env=env)
    working_orders = list(filter(lambda x: x['ordStatus'] == 'Working' and x['accountId'] == accountId, all_orders))
    return working_orders


def get_positions(accountId, env='demo'):
    positions = tv_get('/position/list', {'isAutomated': True}, env=env)
    for account in positions:
        if account['accountId'] == accountId:
            return account
        
    # print("ERROR GETTING POSITIONS")
    print("No positions yet")
    return None

def fill_order(accountId, symbol, quantity, action, currentPrice, order_type='Limit', env='demo'):
    limitPrice = currentPrice + 0.25 if action == "Buy" else currentPrice - 0.25
    endpoint = '/order/placeorder'
    open_order_data = {
        'accountSpec': "sjakes201",
        'accountId': accountId,
        'action': action,
        'symbol': symbol,
        'orderQty': quantity,
        'orderType': order_type,
        'price': limitPrice,
        'isAutomated': True,
    }
    order_response = tv_post(endpoint, open_order_data, env=env)
    order_id = order_response['orderId']
    
    time.sleep(1)
    for i in range(5):
        print(f"Checking for order fill, iteration {i}")
        if i == 4:
            # It has not been filled, cancel the order
            cancel_data = {
                'orderId': order_id, 
                'isAutomated': True
                }
            cancel_response = tv_post('/order/cancelorder', cancel_data, env=env)
            return {
                'result': "UNFILLED AND CANCELLED",
                'response': cancel_response
            }
        else:    
            # Check if it was filled
            print("Checking again if it was filled")
            order_details = get_order_details(order_id)
            print("Order details:", order_details)
            if order_details['ordStatus'] == "Filled":
                return {
                    'result': "SUCCESS",
                    'open_order': {
                        'id': order_id,
                        'limit_price': currentPrice,
                        'direction': action
                    },
                    'symbol': symbol,
                    'quantity': quantity,
                }
            if i < 4:
                time.sleep(2.5)
    return {
        'result': "ERROR"
    }

def open_position(accountId, symbol, quantity, action, currentPrice, order_type='Limit', env='demo'):
    ts_threshold = 10
    
    trailingStopInfo = {
        "Buy": {
            'direction': "Sell",
            'price': currentPrice - ts_threshold
            },
        "Sell": {
            'direction': "Buy",
            'price': currentPrice + ts_threshold
            }
    }.get(action, 0)
    
    limitPrice = currentPrice + 0.25 if action == "Buy" else currentPrice - 0.25
        
    # Place the regular 'to open' order
    endpoint = '/order/placeorder'
    open_order_data = {
        'accountSpec': "sjakes201",
        'accountId': accountId,
        'action': action,
        'symbol': symbol,
        'orderQty': quantity,
        'orderType': order_type,
        'price': limitPrice,
        'isAutomated': True,
    }
    order_response = tv_post(endpoint, open_order_data, env=env)
    order_id = order_response['orderId']
    
    # Continue to check if the order was filled every couple seconds. If the order was filled, submit a trailing stop order. If it was not, cancel the order.
    time.sleep(1)
    for i in range(5):
        print(f"Checking for order fill, iteration {i}")
        if i == 4:
            # It has not been filled, cancel the order
            cancel_data = {
                'orderId': order_id, 
                'isAutomated': True
                }
            cancel_response = tv_post('/order/cancelorder', cancel_data, env=env)
            return {
                'result': "UNFILLED AND CANCELLED",
                'response': cancel_response
            }
        else:    
            # Check if it was filled
            print("Checking again if it was filled")
            order_details = get_order_details(order_id)
            print("Order details:", order_details)
            if order_details['ordStatus'] == "Filled":
                # submit trailing stop order
                trailing_stop_order_data = {
                    'accountSpec': "sjakes201",
                    'accountId': accountId,
                    'action': trailingStopInfo['direction'],
                    'symbol': symbol,
                    'orderQty': quantity,
                    'orderType': "TrailingStop",
                    'stopPrice': trailingStopInfo['price'],
                    'isAutomated': True,
                }
                trailing_stop_data = tv_post(endpoint, trailing_stop_order_data, env=env)
                return {
                    'result': "SUCCESS",
                    'open_order': {
                        'id': order_id,
                        'limit_price': currentPrice,
                        'direction': action
                    },
                    'symbol': symbol,
                    'quantity': quantity,
                    'trailing_stop_order': {
                        'id': trailing_stop_data['orderId'],
                        'stop_price': trailingStopInfo['price'],
                        'direction': trailingStopInfo['direction']
                    }
                }
            if i < 4:
                time.sleep(2.5)
    return {
        'result': "ERROR"
    }
    
def close_position(accountId, symbol, quantity, direction, currentPrice, env='demo'):
    # Submit one position in opposite direction
    # Get list of trailing stops, cancel the one farthest away from current price
    active_trailers = get_active_stops(accountId)
    limitPrice = currentPrice - 0.25 if direction == "Long" else currentPrice + 0.25
    
    ids_list = [x['orderId'] for x in active_trailers]
    trailingstop_ids_to_cancel = ids_list[:quantity]
    
    to_close_action = {"Long": "Sell", "Short": "Buy"}.get(direction)
    
    # Submit order to close position
    endpoint = '/order/placeorder'
    close_order_data = {
        'accountSpec': "sjakes201",
        'accountId': accountId,
        'action': to_close_action,
        'symbol': symbol,
        'orderQty': quantity,
        'orderType': "Limit",
        'price': limitPrice,
        'isAutomated': True,
    }
    order_response = tv_post(endpoint, close_order_data, env=env)
    order_id = order_response['orderId']
    # If successfully closed, cancel a random trailing stop (they should be near each other anyways)
    time.sleep(0.75)
    for i in range(5):
        print(f"Checking for order fill, iteration {i}")
        if i == 4:
            # It has not been filled to close, cancel the order
            cancel_data = {
                'orderId': order_id, 
                'isAutomated': True
                }
            cancel_response = tv_post('/order/cancelorder', cancel_data, env=env)
            return {
                'result': "UNFILLED AND CANCELLED",
                'response': cancel_response
            }
        else:   
            # Check if it was filled
            print("Checking again if it was filled")
            order_details = get_order_details(order_id)
            print("Order details:", order_details)
            if order_details['ordStatus'] == "Filled":
                # Cancel a random trailing stop order
                for i in range(quantity):
                    print(f"Cancelling trailing stop {i} which is orderId {trailingstop_ids_to_cancel[i]}")
                    target_trailing_stop_data = {'orderId': trailingstop_ids_to_cancel[i], 'isAutomated': True}
                    tv_post('/order/cancelorder', target_trailing_stop_data, env=env)
                return {
                    'result': "SUCCESS",
                }
            if i < 4:
                time.sleep(2.5)
    return {
        'result': "ERROR"
    }
    
def get_active_stops(accountId, env='demo'):
    working_orders = get_working_orders(accountId)
    ids_list = [[x['id'], x['action']] for x in working_orders]
    order_versions_list = tv_get('/orderVersion/list', {'isAutomated': True}, env)
    # print("order_versions_list: ", order_versions_list)
    
    # Filter out any ids that are not trailing stop orders
    stop_orders = {item["orderId"] for item in order_versions_list if item.get("orderType") == "Stop"} # TODO: IS THIS RIGHT?
    filtered_ids_list = [[id, data] for id, data in ids_list if id in stop_orders]
    
    # Initialize dictionaries to store highest and lowest stop prices
    lowest_stop_prices = {}
    highest_stop_prices = {}

    # Process each item in ids_list to find highest or lowest stopPrice
    for order_id, action in filtered_ids_list:
        order_id = int(order_id)  # Assuming order_id is converted to int if needed
        if action == 'Buy':
            for order_version in order_versions_list:
                if order_version['orderId'] == order_id:
                    stop_price = order_version['stopPrice']
                    if order_id not in lowest_stop_prices or stop_price < lowest_stop_prices[order_id]:
                        lowest_stop_prices[order_id] = stop_price
        elif action == 'Sell':
            for order_version in order_versions_list:
                if order_version['orderId'] == order_id:
                    stop_price = order_version['stopPrice']
                    if order_id not in highest_stop_prices or stop_price > highest_stop_prices[order_id]:
                        highest_stop_prices[order_id] = stop_price

    # Create a list of dictionaries with action, orderId, and stopPrice
    result = []
    for order_id, action in ids_list:
        if action == 'Buy' and order_id in lowest_stop_prices:
            result.append({'action': 'Buy', 'orderId': order_id, 'stopPrice': lowest_stop_prices[order_id]})
        elif action == 'Sell' and order_id in highest_stop_prices:
            result.append({'action': 'Sell', 'orderId': order_id, 'stopPrice': highest_stop_prices[order_id]})
            
    return result
    
    
"""
strategy is "trailing-stop" or "bracket"
"""
    
def process_signal(accountId, signal, symbol, currentPrice, strategy, config, env='demo'):
    max_position_size = config.get('max_position_size', 7)
    stop_loss_diff = config.get('stop_loss_diff', 4)
    take_profit_diff = config.get('take_profit_diff', 12)
    trailing_loss = config.get('trailing_loss', False)
    
    print(f"Received process_signal for account {accountId}, config {config}, strategy {strategy}, signal {signal} at price {currentPrice} for symbol {symbol}")
    positions = get_positions(accountId, env)
    netPos = 0
    if positions is not None:
        # print(f"ERROR GETTING POSITIONS for account id {accountId}")
        netPos = positions['netPos']
    
    if signal == 'BUY':
        if netPos >= max_position_size:
            # Too large a position, do nothing
            return
        elif netPos >= 0:
            # Submit one long order
            open_bracket_strategy(accountId, symbol, 1, "Buy", currentPrice, take_profit_diff, stop_loss_diff, trailing_loss, env)
        elif netPos == -1:
            # Cancel / close one short, then open one long
            close_result = close_bracket_strategy(accountId, symbol, 1, "Short", currentPrice, env)
            if close_result['result'] == "SUCCESS":
                open_bracket_strategy(accountId, symbol, 1, "Buy", currentPrice, take_profit_diff, stop_loss_diff, trailing_loss, env)
        elif netPos < -1:
            # Cancel two short positions
            close_bracket_strategy(accountId, symbol, 2, "Short", currentPrice, env)
    elif signal == 'SELL':
        if netPos <= -1 * max_position_size:
            # Too large a position, do nothing
            return
        elif netPos <= 0:
            # Submit one short
            open_bracket_strategy(accountId, symbol, 1, "Sell", currentPrice, take_profit_diff, stop_loss_diff, trailing_loss, env)
        elif netPos == 1:
            # Cancel / close one long, then open one short
            close_result = close_bracket_strategy(accountId, symbol, 1, "Long", currentPrice, env)
            if close_result['result'] == "SUCCESS":
                open_bracket_strategy(accountId, symbol, 1, "Sell", currentPrice, take_profit_diff, stop_loss_diff, trailing_loss, env)
        elif netPos > 1:
            # Cancel / close two long positions
            close_bracket_strategy(accountId, symbol, 2, "Long", currentPrice, env)
    else:
        # Signal is HOLD, do nothing
        return
    
    
def get_active_bracket_strategies(accountId, env='demo'):
    strategies = tv_get('/orderStrategy/deps', {"masterid": accountId}, env=env)
    active_strategies = [strategy for strategy in strategies if strategy['status'] == 'ActiveStrategy']
    return active_strategies
    
def open_bracket_strategy(accountId, symbol, quantity, action, currentPrice, take_profit_distance, stop_loss_diff, trailing_loss, env='demo'):

    closing_action = {"Buy": "Sell", "Sell": "Buy"}.get(action)
    oso_orders = oso_orders_from_acc_id.get(accountId)
    
    if action == "Sell":
        take_profit_distance = -take_profit_distance
        
    take_profit = {
        "action": closing_action,
        "orderType": "Limit",
        "price": currentPrice + take_profit_distance
    }
    
    if action == "Buy":
        stop_loss_diff = -stop_loss_diff
        
    if trailing_loss:
        orderType = "TrailingStop"
    else:
        orderType = "Stop"
        
    stop_loss = {
        "action": closing_action,
        "orderType": orderType,
        "stopPrice": currentPrice + stop_loss_diff
    }
    
    limit_price = currentPrice + 0.25 if action == "Buy" else currentPrice - 0.25
    
    data = {
        'accountSpec': "sjakes201",
        'accountId': accountId,
        'action': action,
        'symbol': symbol,
        'orderQty': quantity,
        'orderType': "Limit",
        'price': limit_price,
        'isAutomated': True,
        'bracket1': take_profit,
        'bracket2': stop_loss
    }
    
    oso_data = tv_post('/order/placeoso', data, env=env)
    orderId: int = oso_data['orderId']
    time.sleep(5)
    order_details = get_order_details(orderId, env=env)
    if order_details['ordStatus'] != "Filled":
        tv_post('/order/cancelorder', {"orderId": orderId}, env=env)
    elif order_details['ordStatus'] == "Filled":
        oso_data['direction'] = {
            "Buy": "Long",
            "Sell": "Short"
        }.get(action)
        oso_orders.append(oso_data)
    else:
        print("Error in opening bracket strategy, uncertain order fill status: ", order_details)

def close_bracket_strategy(accountId, symbol, quantity, direction, currentPrice, env='demo'):
    oso_orders = oso_orders_from_acc_id.get(accountId)
    
    # Get active bracket strategies that can be cancelled by action
    current_brackets = [strategy for strategy in oso_orders if strategy['direction'] == direction]
    if len(current_brackets) < quantity:
        print(f"Not enough stored strategies to cancel, oso_orders has {len(oso_orders)} items, but only {len(current_brackets)} are {direction} and quantity is {quantity}")
        # Problem: how is it not stored? Did the algorithm restart?
        return {
            "result": "Not enough stored strategies to cancel"
        }

    direction_to_close = {
        "Long": "Sell",
        "Short": "Buy"
    }.get(direction)
    
    # First, cancel a position
    close_result = fill_order(accountId, symbol, quantity, direction_to_close, currentPrice, env=env)
    if close_result["result"] != "SUCCESS":
        # Failed to close position, will not close brackets
        print("Failed to close position")
        return {
            "result": "Failed to close position"
        }
    
    # Second, cancel the brackets
    for i in range(quantity):
        to_cancel = current_brackets[i]
        oso_orders.pop(0)
        tv_post('/order/cancelorder', {"orderId": to_cancel["oso1Id"]}, env=env)
        tv_post('/order/cancelorder', {"orderId": to_cancel["oso2Id"]}, env=env)
        time.sleep(0.5)
    return {
        "result": "SUCCESS"
    }

def init_connection():
    connect(credentials)
    # Retrieve and handle account list
    accounts = tv_get('/account/list')
    handle_account_list(accounts)