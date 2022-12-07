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
            N_passenger = result['passengers']
            try:
                N_passenger = int(N_passenger.replace(',', ''))
            except ValueError:
                pass
            else:
                if N_passenger >= passengerCount:
                    Ship_List.append(result['name'])
        url = Data['next']

    return Ship_List
