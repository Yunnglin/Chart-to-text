# encoding:utf-8
import requests

def get_access_token():
    # client_id 为官网获取的AK， client_secret 为官网获取的SK
    api_key = 'RTiX6wDKA3gkm01Bm2S0l0xO'
    secret_key = 'gEjMhTtKFKb9xTm4B8RHzwiMliTq8M8B'
    host = f'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={api_key}&client_secret={secret_key}'
    response = requests.get(host)
    if response:
        print(response.json())

if __name__=='__main__':
    get_access_token()