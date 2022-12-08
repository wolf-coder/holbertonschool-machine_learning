#!/usr/bin/env python3
"""
script that prints the location of a specific user
"""
import sys
from time import time as time_Since_Epoch
import requests


if __name__ == '__main__':
    url = sys.argv[1]
    response = requests.get(url)
    if response.status_code == 404:
        print("Not found")
    elif response.status_code == 403:
        Ratelimit = int(response.headers['X-Ratelimit-Reset'])
        current = int(time_Since_Epoch())
        X = int((Ratelimit - current) / 60)
        print("Reset in {} min".format(int(X)))
    else:
        response = response.json()
        print(response['location'])
