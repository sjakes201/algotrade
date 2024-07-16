import time
import requests
from datetime import datetime, timedelta
from services import tv_get, tv_post
from storage import set_access_token, get_access_token, set_available_accounts, token_is_valid
from utils import wait_for_ms

def handle_retry(data, response):
    ticket = response['p-ticket']
    penalty_time = response['p-time']
    captcha = response.get('p-captcha')

    if captcha:
        raise Exception("Captcha present, cannot retry auth request via third-party application. Please try again in an hour.")

    print(f"Time Penalty present. Retrying operation in {penalty_time} seconds")
    wait_for_ms(penalty_time * 1000)
    return connect({**data, 'p-ticket': ticket})

def connect(data):
    token, expiration = get_access_token()

    if token and token_is_valid(expiration):
        print("Already connected. Using valid token.")
        accounts = tv_get('/account/list')
        set_available_accounts(accounts)
        return

    auth_response = tv_post('/auth/accesstokenrequest', data, use_token=False)

    if 'p-ticket' in auth_response:
        return handle_retry(data, auth_response)
    else:
        if 'errorText' in auth_response:
            raise Exception(auth_response['errorText'])

        access_token = auth_response['accessToken']
        expiration_time = auth_response['expirationTime']
        set_access_token(access_token, expiration_time)

        accounts = tv_get('/account/list')
        set_available_accounts(accounts)

        print(f"Successfully stored Tradovate access token {access_token}.")
