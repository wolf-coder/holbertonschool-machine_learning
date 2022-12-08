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
    Response = requests.get("https://api.spacexdata.com/v4/launches/")
    rockets = {}
    DATA = Response.json()
    for launch in DATA:
        rocket = launch["rocket"]
        rocket_url = "https://api.spacexdata.com/v4/rockets/" + rocket
        r_rocket = requests.get(rocket_url)
        r_rocket_get = r_rocket.json()
        rocket_name = r_rocket_get["name"]
        if rocket_name in rockets.keys():
            rockets[rocket_name] = rockets[rocket_name] + 1
        else:
            rockets[rocket_name] = 1
    Sorted = sorted(rockets.items(), key=lambda x: x[0])
    Sorted = sorted(Sorted, key=lambda x: x[1], reverse=True)
    for i in sorting:
        print("{}: {}".format(i[0], i[1]))
