#!/usr/bin/env python3
"""
By using the (unofficial) SpaceX API, write a script that displays the number of launches per rocket.
  + All launches should be taking in consideration
  + Each line should contain the rocket name and the number of
launches separated by`:`
  + Order the result by the number launches (descending)
  + If multiple rockets have the same amount of launches, order them by alphabetic order (A to Z)
"""

import requests
import time


if __name__ == '__main__':
    rockets = {}
    launches_req = requests.get(
        'https://api.spacexdata.com/v4/launches'
    ).json()
    for result in launches_req:
        rocket_id = result['rocket']
        if rocket_id in rockets:
            rockets[rocket_id] += 1
        else:
            rockets[rocket_id] = 1
    rockets = sorted(
        rockets.items(), key=lambda rocket: rocket[1], reverse=True
    )
    for rocket in rockets:
        rocket_name = requests.get(
            'https://api.spacexdata.com/v4/rockets/' + rocket[0]
        ).json()['name']
        print('{}: {}'.format(
            rocket_name,
            rocket[1]
        ))
