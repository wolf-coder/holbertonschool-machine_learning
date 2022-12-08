#!/usr/bin/env python3
"""
Using the (unofficial) SpaceX API, write a script that displays
 the upcoming launch with these information:
  + Name of the launch
  + The date (in local time)
  + The rocket name
  + The name (with the locality) of the launchpad
"""
import requests


if __name__ == '__main__':

    Response = requests.get(
        'https://api.spacexdata.com/v4/launches/upcoming').json()

    # Sorting the list in term of date
    Response.sort(key=lambda json: json['date_unix'])

    upcoming = Response[0]
    launch_date = upcoming['date_local']
    launch_name = upcoming['name']

    rocket_name = requests.get(
        'https://api.spacexdata.com/v4/rockets/' + upcoming['rocket']
    ).json()['name']
    launchpad = requests.get(
        'https://api.spacexdata.com/v4/launchpads/' + upcoming['launchpad']
    ).json()

    launchpad_locality = launchpad['locality']
    launchpad_name = launchpad['name']
    print('{} ({}) {} - {} ({})'.format(
        launch_name,
        launch_date,
        rocket_name,
        launchpad_name,
        launchpad_locality
    ))
