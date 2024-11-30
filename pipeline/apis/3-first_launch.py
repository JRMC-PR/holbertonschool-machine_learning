#!/usr/bin/env python3
import requests
from datetime import datetime
import pytz

def get_first_launch():
    """
    Fetch and display the first SpaceX launch with the following information:
    - Name of the launch
    - The date (in local time)
    - The rocket name
    - The name (with the locality) of the launchpad
    Format: <launch name> (<date>) <rocket name> - <launchpad name> (<launchpad locality>)
    """
    # SpaceX API URL for launches
    url = "https://api.spacexdata.com/v4/launches"

    # Send a GET request to the SpaceX API
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch data: {response.status_code}")

    # Parse the JSON response
    launches = response.json()

    # Sort launches by date_unix
    launches.sort(key=lambda x: x['date_unix'])

    # Get the first launch
    first_launch = launches[0]

    # Extract launch details
    launch_name = first_launch["name"]
    launch_date = datetime.fromtimestamp(first_launch["date_unix"], pytz.utc).astimezone().isoformat()
    rocket_id = first_launch["rocket"]
    launchpad_id = first_launch["launchpad"]

    # Fetch rocket details
    rocket_url = f"https://api.spacexdata.com/v4/rockets/{rocket_id}"
    rocket_response = requests.get(rocket_url)
    rocket_name = rocket_response.json()["name"]

    # Fetch launchpad details
    launchpad_url = f"https://api.spacexdata.com/v4/launchpads/{launchpad_id}"
    launchpad_response = requests.get(launchpad_url)
    launchpad_data = launchpad_response.json()
    launchpad_name = launchpad_data["name"]
    launchpad_locality = launchpad_data["locality"]

    # Format and display the launch information
    print(f"{launch_name} ({launch_date}) {rocket_name} - {launchpad_name} ({launchpad_locality})")

if __name__ == "__main__":
    get_first_launch()
