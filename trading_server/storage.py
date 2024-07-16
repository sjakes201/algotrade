from datetime import datetime
import json

STORAGE = {
    "access_token": None,
    "expiration": None,
    "available_accounts": None,
}

def set_access_token(token, expiration):
    STORAGE["access_token"] = token
    STORAGE["expiration"] = expiration

def get_access_token():
    return STORAGE["access_token"], STORAGE["expiration"]

def set_available_accounts(accounts):
    STORAGE["available_accounts"] = accounts

def get_available_accounts():
    return STORAGE["available_accounts"]

def token_is_valid(expiration):
    # Temporarily return False since we want to renew before expiration
    return False
    return datetime.strptime(expiration, '%Y-%m-%dT%H:%M:%SZ') > datetime.utcnow()
