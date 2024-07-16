import requests
from storage import get_access_token
import time

BASE_URLS = {
    'demo': 'https://demo-api.tradovate.com/v1',
    'live': 'https://live-api.tradovate.com/v1'
}

def tv_get(endpoint, query=None, env='live'):
    token, _ = get_access_token()
    base_url = BASE_URLS[env]

    url = base_url + endpoint
    if query:
        url += '?' + '&'.join([f"{key}={value}" for key, value in query.items()])

    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/json',
        'Content-Type': 'application/json'
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    return response.json()

def tv_post(endpoint, data, use_token=True, env='live'):
    token, _ = get_access_token()
    base_url = BASE_URLS[env]

    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
    }
    if use_token:
        headers['Authorization'] = f'Bearer {token}'

    response = requests.post(base_url + endpoint, json=data, headers=headers)
    response.raise_for_status()

    return response.json()    
    

def get_account_balance(account_id, env='live'):
    endpoint = f'/account/balance/{account_id}'
    return tv_get(endpoint, env=env)
