
## Data Collection - APIs


# Python Requests Package Guide

## 1. How to Use the Python Package `requests`
The `requests` package is a popular Python library for making HTTP requests. It simplifies interacting with APIs and web resources.

### Installation:
```bash
pip install requests
```

### Basic Usage:
```python
import requests

response = requests.get("https://example.com")
print(response.status_code)  # HTTP status code
print(response.text)         # Response body as text
```

---

## 2. How to Make an HTTP GET Request
An HTTP GET request retrieves data from a server. The `requests.get()` method handles this in Python.

### Example:
```python
import requests

url = "https://api.example.com/data"
response = requests.get(url)

if response.status_code == 200:
    print("Data retrieved:", response.json())  # Convert JSON response to a Python dictionary
else:
    print("Failed to fetch data:", response.status_code)
```

### Adding Query Parameters:
```python
params = {"key1": "value1", "key2": "value2"}
response = requests.get(url, params=params)
```

---

## 3. How to Handle Rate Limits
Rate limiting is used by APIs to restrict the number of requests over a time frame. Handling rate limits ensures compliance with API rules.

### Example:
```python
import requests
import time

url = "https://api.example.com/data"
headers = {"Authorization": "Bearer your_token"}

while True:
    response = requests.get(url, headers=headers)

    if response.status_code == 429:  # Too Many Requests
        retry_after = int(response.headers.get("Retry-After", 1))
        print(f"Rate limited. Retrying in {retry_after} seconds...")
        time.sleep(retry_after)
    elif response.status_code == 200:
        print("Success:", response.json())
        break
    else:
        print("Error:", response.status_code)
        break
```

---

## 4. How to Handle Pagination
Many APIs return large datasets in pages. Handling pagination ensures you retrieve all available data.

### Example:
```python
import requests

url = "https://api.example.com/data"
data = []

while url:
    response = requests.get(url)
    if response.status_code == 200:
        page_data = response.json()
        data.extend(page_data["results"])  # Append results
        url = page_data.get("next")       # Get next page URL
    else:
        print("Error fetching data:", response.status_code)
        break

print("All data:", data)
```

---

## 5. How to Fetch JSON Resources
JSON (JavaScript Object Notation) is a common format for API responses. Use `response.json()` to parse it.

### Example:
```python
import requests

url = "https://api.example.com/data"
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    print(data)  # JSON parsed as a Python dictionary
else:
    print("Failed to fetch JSON:", response.status_code)
```

---

## 6. How to Manipulate Data from an External Service
After fetching data, you can manipulate it using Python's data structures.

### Example:
```python
import requests

url = "https://api.example.com/users"
response = requests.get(url)

if response.status_code == 200:
    users = response.json()  # List of user data
    # Example: Extract names of users older than 30
    older_users = [user["name"] for user in users if user["age"] > 30]
    print("Users older than 30:", older_users)
else:
    print("Error:", response.status_code)
```

---

### Summary
The `requests` library makes it simple to:
- Make HTTP requests (GET, POST, etc.).
- Handle common API tasks like rate limits and pagination.
- Parse and manipulate JSON data.
By mastering these techniques, you can efficiently interact with external services.


### Description
0. Can I join?mandatoryBy using theSwapi API, create a method that returns the list of ships that can hold a given number of passengers:Prototype:def availableShips(passengerCount):Don’t forget the paginationIf no ship available, return an empty list.bob@dylan:~$ cat 0-main.py
#!/usr/bin/env python3
"""
Test file
"""
availableShips = __import__('0-passengers').availableShips
ships = availableShips(4)
for ship in ships:
    print(ship)

bob@dylan:~$ ./0-main.py
CR90 corvette
Sentinel-class landing craft
Death Star
Millennium Falcon
Executor
Rebel transport
Slave 1
Imperial shuttle
EF76 Nebulon-B escort frigate
Calamari Cruiser
Republic Cruiser
Droid control ship
Scimitar
J-type diplomatic barge
AA-9 Coruscant freighter
Republic Assault ship
Solar Sailer
Trade Federation cruiser
Theta-class T-2c shuttle
Republic attack cruiser
bob@dylan:~$Repo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/apisFile:0-passengers.pyHelp×Students who are done with "0. Can I join?"Review your work×Correction of "0. Can I join?"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/5pts

1. Where I am?mandatoryBy using theSwapi API, create a method that returns the list of names of the home planets of allsentientspecies.Prototype:def sentientPlanets():Don’t forget the paginationsentienttype is either in theclassificationordesignationattributes.bob@dylan:~$ cat 1-main.py
#!/usr/bin/env python3
"""
Test file
"""
sentientPlanets = __import__('1-sentience').sentientPlanets
planets = sentientPlanets()
for planet in planets:
    print(planet)

bob@dylan:~$ ./1-main.py
Endor
Naboo
Coruscant
Kamino
Geonosis
Utapau
Kashyyyk
Cato Neimoidia
Rodia
Nal Hutta
unknown
Trandosha
Mon Cala
Sullust
Toydaria
Malastare
Ryloth
Aleen Minor
Vulpter
Troiken
Tund
Cerea
Glee Anselm
Iridonia
Tholoth
Iktotch
Quermia
Dorin
Champala
Mirial
Zolan
Ojom
Skako
Muunilinst
Shili
Kalee
bob@dylan:~$Repo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/apisFile:1-sentience.pyHelp×Students who are done with "1. Where I am?"Review your work×Correction of "1. Where I am?"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/4pts

2. Rate me is you can!mandatoryBy using theGitHub API, write a script that prints the location of a specific user:The user is passed as first argument of the script with the full API URL, example:./2-user_location.py https://api.github.com/users/holbertonschoolIf the user doesn’t exist, printNot foundIf the status code is403, printReset in X minwhereXis the number of minutes from now and the value ofX-Ratelimit-ResetYour code should not be executed when the file is imported (you should useif __name__ == '__main__':)bob@dylan:~$ ./2-user_location.py https://api.github.com/users/Holbertonschoolml
San Francisco, CA
bob@dylan:~$
bob@dylan:~$ ./2-user_location.py https://api.github.com/users/holberton_ho_no
Not found
bob@dylan:~$
... after a lot of requests ... 60 exactly...
bob@dylan:~$
bob@dylan:~$ ./2-user_location.py https://api.github.com/users/Holbertonschoolml
Reset in 16 min
bob@dylan:~$Tips:Playing with an API that has a Rate limit is challenging, mainly because you don’t have the control on when the quota will be reset - we really encourage you to analyze the API a much as you can before coding and be able to “mock the API response”Repo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/apisFile:2-user_location.pyHelp×Students who are done with "2. Rate me is you can!"Review your work×Correction of "2. Rate me is you can!"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/6pts

3. First launchmandatoryBy using the(unofficial) SpaceX API, write a script that displays the first launch with these information:Name of the launchThe date (in local time)The rocket nameThe name (with the locality) of the launchpadFormat:<launch name> (<date>) <rocket name> - <launchpad name> (<launchpad locality>)we encourage you to use thedate_unixfor sorting it - and if 2 launches have the same date, use the first one in the API result.Your code should not be executed when the file is imported (you should useif __name__ == '__main__':)bob@dylan:~$ ./3-upcoming.py
Galaxy 33 (15R) & 34 (12R) (2022-10-08T19:05:00-04:00) Falcon 9 - CCSFS SLC 40 (Cape Canaveral)
bob@dylan:~$Repo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/apisFile:3-first_launch.pyHelp×Students who are done with "3. First launch"Review your work×Correction of "3. First launch"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/4pts

4. How many by rocket?mandatoryBy using the(unofficial) SpaceX API, write a script that displays the number of launches per rocket.Use thishttps://api.spacexdata.com/v4/launchesto makerequestAll launches should be taken into considerationEach line should contain the rocket name and the number of launches separated by:(format below in the example)Order the result by the number launches (descending)If multiple rockets have the same amount of launches, order them by alphabetic order (A to Z)Your code should not be executed when the file is imported (you should useif __name__ == '__main__':)bob@dylan:~$ ./4-rocket_frequency.py
Falcon 9: 103
Falcon 1: 5
Falcon Heavy: 3
bob@dylan:~$Repo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/apisFile:4-rocket_frequency.pyHelp×Students who are done with "4. How many by rocket?"Review your work×Correction of "4. How many by rocket?"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/4pts

**Repository:**
- GitHub repository: `holbertonschool-machine_learning`
- Directory: `supervised_learning/classification`
- File: `Data_Collection__APIs.md`
