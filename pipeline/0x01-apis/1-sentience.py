#!/usr/bin/env python3
"""
Exploiting the Species resouce
"""
import requests as RE


def sentientPlanets():
    """
    Method that returns the list of names of the home planets
 of all sentient species.
      + Prototype: def sentientPlanets():
    """
    Sentient_planets = []
    url = "https://swapi-api.hbtn.io/api/species/"
    while (url):
        Request = RE.get(url)
        DATA = Request.json()
        for result in DATA['results']:
            if result['designation'] == 'sentient' or\
               result["designation"] == 'sentient':
                #breakpoint()
                if result['homeworld']:
                    Home_world = RE.get(result['homeworld']).json()['name']
                else:
                    Home_world = "unknown"
                if Home_world not in Sentient_planets:
                    Sentient_planets.append(Home_world)
        url = DATA['next']
    return Sentient_planets
