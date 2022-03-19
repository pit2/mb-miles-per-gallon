import requests


def print_forecast(forecast):
    weather = forecast["weather_state_name"]
    min_temp, max_temp = round(
        forecast["min_temp"]), round(forecast["max_temp"])
    print(f"{weather} with temperatures between {min_temp} °C and {max_temp} °C")
    wind_dir = forecast["wind_direction_compass"]
    wind_speed = round(forecast["wind_speed"], 0)
    print(f"Wind speed {wind_speed} mps from {wind_dir}")
    air_pressure, humidity = round(
        forecast["air_pressure"]), round(forecast["humidity"])
    print(f"Air pressure: {air_pressure} mbar")
    print(f"Humidity: {humidity}%")
    visibility = round(forecast["visibility"])
    print(f"Visibility: {visibility} miles")


params = {"query": "leipzig"}
headers = {"Content-Type": "application/json"}
result = requests.get("https://www.metaweather.com/api/location/search/",
                      params=params, headers=headers)

json = result.json()
woeid = json[0]["woeid"]
print("woeid: ", woeid)  # 671072
print("Lattitude, longitude: ", json[0]["latt_long"])  # 51.3452,12.38594


result = requests.get(
    "https://www.metaweather.com/api/location/"+str(woeid), headers=headers)
json = result.json()
print("Forecast for ", json["title"])
forecast = json["consolidated_weather"]
for i in range(5):
    print("On ", forecast[i]["applicable_date"])
    print_forecast(forecast[i])

result = requests.get("https://www.metaweather.com/api/location/" +
                      str(woeid)+"/2019/3/8/", headers=headers)
print("Weather for Leipzig on 8th March 2019:")
print_forecast(result.json()[0])
