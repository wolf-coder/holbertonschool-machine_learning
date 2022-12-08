#!/usr/bin/env python3
"""
using the (unofficial) SpaceX API, write a script that displays the upcoming launch with these information:
  + Name of the launch
  + The date (in local time)
  + The rocket name
  + The name (with the locality) of the launchpad
"""
import requests


if __name__ == '__main__':
    url = 'https://api.spacexdata.com/v4/rockets'
    rockets = [
        rocket['name'] for rocket in requests.get(url).json()
    ]
    url = 'https://api.spacexdata.com/v3/launches'
    for rocket in rockets:
        params = {'rocket_name': rocket}
        print('{}: {}'.format(rocket, len(requests.get(url, params=params).json())
        ))
