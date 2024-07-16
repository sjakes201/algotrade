import pytest
from Account import Account
from Position import Position

@pytest.fixture(scope="function")
def account():
    algo_params = {
        'confidence_threshold': 0.5,
        'take_profit_diff': 10,
        'stop_loss_diff': 9,
        'max_position_size': 6
    }
    acc = Account("test_account", 10000, algo_params)
    acc.set_contract_config(per_contract_fee=0.35, point_dollar_value=5, points_slippage=0)
    return acc

def test_order_to_open(account):
    account.order_to_open("ESU4", "LONG", 150, "2024-07-04 10:00:00")
    assert len(account.get_positions()) == 1
    assert account.get_positions()[0].symbol == "ESU4"
    assert account.get_positions()[0].type == "LONG"
    assert account.get_positions()[0].entry_price == 150
    assert account.get_positions()[0].stop_loss['price'] == 141
    assert account.get_positions()[0].take_profit['price'] == 160
    
def test_small_profit(account):
    account.order_to_open("ESU4", "LONG", 100, "2024-07-04 10:00:00")
    assert account.balance == 10000 - (1*0.35)
    account.order_to_close("ESU4", "LONG", 104)
    assert len(account.get_positions()) == 0
    assert account.balance == 10000 + (4*5) - (2*0.35)
    
def test_take_profit(account):
    account.order_to_open("ESU4", "LONG", 100, "2024-07-04 10:00:00")
    account.order_to_close("ESU4", "LONG", 125)
    account.order_to_open("ESU4", "SHORT", 127, "2024-07-04 10:00:00")
    account.order_to_close("ESU4", "SHORT", 120)
    approx_balance = 10000 + (20*5) - (4*0.35)
    assert account.balance == pytest.approx(approx_balance, 0.01)

def test_small_loss(account):
    account.order_to_open("ESU4", "LONG", 100, "2024-07-04 10:00:00")
    account.order_to_close("ESU4", "LONG", 96)
    account.order_to_open("ESU4", "SHORT", 100, "2024-07-04 10:00:00")
    account.order_to_close("ESU4", "SHORT", 103)
    approx_balance = 10000 - (7*5) - (2*0.35)
    assert account.balance == pytest.approx(approx_balance, 0.01)
    
def test_stop_loss(account):
    account.order_to_open("ESU4", "LONG", 100, "2024-07-04 10:00:00")
    account.order_to_close("ESU4", "LONG", 84)
    approx_balance = 10000 - (9*5) - (2*0.35)
    assert account.balance == pytest.approx(approx_balance, 0.01)
    
def test_both_directions(account):
    account.order_to_open("ESU4", "SHORT", 100, "2024-07-04 10:00:00")
    account.order_to_close("ESU4", "SHORT", 97)
    account.order_to_open("ESU4", "LONG", 92, "2024-07-04 10:00:00")
    account.order_to_close("ESU4", "LONG", 94.5)
    approx_balance = 10000 + (5.5*5) - (4*0.35)
    assert account.balance == pytest.approx(approx_balance, 0.01)
    
def test_multi_leg(account):
    account.order_to_open("ESU4", "LONG", 100, "2024-07-04 10:00:00")
    account.order_to_open("ESU4", "LONG", 103.25, "2024-07-04 10:00:00")
    account.order_to_open("ESU4", "LONG", 102.5, "2024-07-04 10:00:00")
    account.order_to_close("ESU4", "LONG", 98.5)
    account.order_to_open("ESU4", "LONG", 99.75, "2024-07-04 10:00:00")
    account.order_to_close("ESU4", "LONG", 103.5)
    account.order_to_close("ESU4", "LONG", 105)
    account.order_to_close("ESU4", "LONG", 107.25)
    approx_balance = 10000 + 10.9375 - (8*0.35)
    assert account.balance == pytest.approx(approx_balance, 0.01)
    
def test_close_all(account):
    account.order_to_open("ESM4", "LONG", 100, "2024-07-04 10:00:00")
    account.order_to_open("ESM4", "LONG", 105, "2024-07-04 10:00:00")
    account.close_all_positions("ESM4", 114)
    approx_balance =  10000 + (10+9)*(5) - (4*0.35)
    assert account.balance == pytest.approx(approx_balance, 0.01)
    
def test_check_all_positions(account):
    account.order_to_open("ESU3", "SHORT", 98, "2024-07-04 10:00:00")
    account.order_to_open("ESU3", "SHORT", 105, "2024-07-04 10:00:00")
    account.check_all_positions("ESM4", 110)
    approx_balance = 10000 + (9)*5 - (3*0.35)
    assert account.balance == pytest.approx(approx_balance, 0.01)
    