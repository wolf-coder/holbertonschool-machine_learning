#!/usr/bin/env python3
"""
Starships resources
"""
import requests as RE


def availableShips(passengerCount):
    """
    - method that returns the list of ships that can hold a given number
 of passengers:
      + If no ship available, return an empty list.
    """

    Ship_List = []

    # The URL root for Starships resources
    url = "https://swapi-api.hbtn.io/api/starships/"

    while (url):
        Request = RE.get(url)
        Data = Request.json()
        for result in Data['results']:
            try:
                N_passenger = result['passengers']
                N_passenger = ''.join(N_passenger.split(','))
                N_passenger = int(N_passenger)
            except ValueError:
                pass
            else:
                Ship_List.append(result['name'])
        url = Data['next']

    return Ship_List
