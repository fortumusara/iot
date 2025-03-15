# Imports
import requests
import pandas as pd

# Download data from URL
res = requests.get(URL)

# Convert the result
data_temp = res.json()
print(data_temp)

# Convert json data to DataFrame
df_temp = pd.DataFrame(data_temp)

print(df_temp.head())
----------------------------------------------------------------------
# Import pandas
import pandas as pd

# Load URL to DataFrame
df_temp = pd.read_json(URL)

# Print first 5 rows
print(df_temp.head())

# Print datatypes
print(df_temp.dtypes)
----------------------------------------------------------------------
df.to_json("abc.json", orient="records")
----------------------------------------------------------------------
# Import pandas
import pandas as pd

# Load URL to DataFrame
df_temp = pd.read_json(URL)

# Save DataFrame as json
df_temp.to_json("temperature.json", orient="records")

# Save DataFrame as csv without index
df_temp.to_csv("temperature.csv", index=False)
----------------------------------------------------------------------
import pandas as pd

# Read file
df_env = pd.read_csv("environmental.csv", parse_dates=["timestamp"])

# Print head
print(df_env.head())

# Print DataFrame info
print(df_env.info())

------------------------------------------------------------------------
import pandas as pd

# Read file
df_env = pd.read_json("environmental.json")

# Print head
print(df_env.head())

# Print DataFrame info
print(df_env.info())
------------------------------------------------------------------------
import pandas as pd

# Read file from json
df_env = pd.read_json("environmental.json")

# Print summary statistics
print(df_env.describe())
-----------------------------------------------------------------------
# Import mqtt library
import paho.mqtt.subscribe as subscribe

# Retrieve one message
msg = subscribe.simple("datacamp/iot/simple", hostname="mqtt.datacamp.com")

# Print topic and payload
print(f"{msg.topic}, {msg.payload}")

---------------------------------------------------------------------------
# Save Datastream
# You will now take an MQTT Data stream and append each new data point to the list store.

# Using the library paho.mqtt, you can subscribe to a data stream using subscribe.callback()
# Define function to call by callback method
def on_message(client, userdata, message):
    # Parse the message.payload
    data = json.loads(message.payload)
    store.append(data)

# Connect function to mqtt datastream
subscribe.callback(on_message, topic, MQTT_HOST)

df = pd.DataFrame(store)
print(df.head())

# Store DataFrame to csv, skipping the index
df.to_csv("datastream.csv", index=False)
-----------------------------------------------------------------------------------------------
cols = ["temperature", "humidity", "pressure"]

# Create a line plot
df[cols].plot(title="Environmental data")

# Label X-Axis
plt.xlabel("Time")

# Show plot
plt.show()
-----------------------------------------------------------------------------------------------
cols = ["temperature", "humidity", "pressure"]

# Create a line plot
df[cols].plot(title="Environmental data",
              secondary_y="pressure")

# Label X-Axis
plt.xlabel("Time")

# Show plot
plt.show()
----------------------------------------------------------------------------------------------
#Create Histograms
cols = ["temperature", "humidity", "pressure", "radiation"]

# Create a histogram
df[cols].hist(bins=30)

# Label Y-Axis
plt.ylabel("frequency")

# Show plot
plt.show()
-----------------------------------------------------------------------------------------------
#Dealing with null values
# Print head of the DataFrame
print(data.head())

# Drop missing rows
data_clean = data.dropna()
print(data_clean.head())
-----------------------------------------------------------------------------------------------
# Print head of the DataFrame
print(data.head())

# Forward-fill missing values
data_clean = data.ffill()
print(data_clean.head())
-----------------------------------------------------------------------------------------------
# Calculate and print NA count
print(data.isna().sum())
----------------------------------------------------------------------------------------------
# Identify intervals with no data by resampling in intervals

# Calculate and print the sum of NA values
print(data.isna().sum())

# Resample data
data_res = data.resample("10min").last()

# Calculate and print NA count
print(data_res.isna().sum())
-----------------------------------------------------------------------------------------------
a Django API that retrieves and analyzes stored fire alarm data. The API will:

✅ Ingest data from CSV (or MQTT directly in the future)
✅ Expose endpoints to query sensor data
✅ Provide alerts based on threshold values (e.g., high smoke levels, extreme temperatures)
✅ Integrate with AWS S3 for storage (optional for scalability)
-----------------------------------------------------------------------------------------------

from django.shortcuts import get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import os

# Load data from CSV (simulate a database for now)
CSV_FILE = "datastream.csv"

def load_data():
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE)
    return pd.DataFrame(columns=["sensor_id", "temperature", "smoke_level", "timestamp"])

data = load_data()

# API Endpoint to get all sensor data
@csrf_exempt
def get_sensor_data(request):
    global data
    data_dict = data.to_dict(orient='records')
    return JsonResponse({"sensors": data_dict}, safe=False)

# API Endpoint to get a specific sensor data
@csrf_exempt
def get_sensor_by_id(request, sensor_id):
    global data
    sensor_data = data[data["sensor_id"] == sensor_id]
    if sensor_data.empty:
        return JsonResponse({"error": "Sensor not found"}, status=404)
    return JsonResponse(sensor_data.to_dict(orient='records'), safe=False)

# API Endpoint for fire alert detection
@csrf_exempt
def get_fire_alerts(request):
    global data
    alert_data = data[(data["temperature"] > 80) | (data["smoke_level"] > 0.04)]
    return JsonResponse({"alerts": alert_data.to_dict(orient='records')}, safe=False)

# URL Patterns (To be added in urls.py)
from django.urls import path
urlpatterns = [
    path('api/sensors/', get_sensor_data, name='get_sensor_data'),
    path('api/sensors/<str:sensor_id>/', get_sensor_by_id, name='get_sensor_by_id'),
    path('api/alerts/', get_fire_alerts, name='get_fire_alerts'),
]
--------------------------------------------------------------------------------------------------
This Django API now:

✅ Consumes MQTT messages and stores them in a database
✅ Provides REST endpoints for sensor data retrieval
✅ Alerts when temperature/smoke thresholds exceed limits
--------------------------------------------------------------------------------------------------
import json
import paho.mqtt.client as mqtt
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import SensorData
from .serializers import SensorDataSerializer

# MQTT Configuration
MQTT_BROKER = "your-mqtt-broker"
MQTT_TOPIC = "fire_alarm/data"

def on_message(client, userdata, msg):
    """ Callback function to handle incoming MQTT messages. """
    try:
        data = json.loads(msg.payload)
        SensorData.objects.create(
            sensor_id=data["sensor_id"],
            temperature=data["temperature"],
            smoke_level=data["smoke_level"],
            timestamp=data["timestamp"]
        )
    except Exception as e:
        print(f"Error processing message: {e}")

# MQTT Client Setup
client = mqtt.Client()
client.on_message = on_message
client.connect(MQTT_BROKER, 1883, 60)
client.subscribe(MQTT_TOPIC)
client.loop_start()

@api_view(['GET'])
def get_sensors(request):
    """ API endpoint to retrieve all sensor data """
    sensors = SensorData.objects.all()
    serializer = SensorDataSerializer(sensors, many=True)
    return Response(serializer.data)

@api_view(['GET'])
def get_sensor_by_id(request, sensor_id):
    """ API endpoint to retrieve sensor data by ID """
    sensor = get_object_or_404(SensorData, sensor_id=sensor_id)
    serializer = SensorDataSerializer(sensor)
    return Response(serializer.data)

@api_view(['GET'])
def get_fire_alerts(request):
    """ API endpoint to retrieve alerts for high temperature/smoke levels """
    alerts = SensorData.objects.filter(temperature__gt=60, smoke_level__gt=5)
    serializer = SensorDataSerializer(alerts, many=True)
    return Response(serializer.data, status=status.HTTP_200_OK)

# Add URLs in urls.py
# path('api/sensors/', get_sensors),
# path('api/sensors/<int:sensor_id>/', get_sensor_by_id),
# path('api/alerts/', get_fire_alerts),

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a datetime range
time_index = pd.date_range(start="2025-03-01 08:00", periods=10, freq="5min")

# Generate temperature data with some missing values
temperature = [22.5, np.nan, 23.1, np.nan, 24.0, np.nan, 25.2, np.nan, 26.5, 27.0]
humidity = [60, 65, np.nan, 68, np.nan, 72, np.nan, 74, 75, np.nan]

# Create a DataFrame
data = pd.DataFrame({"temperature": temperature, "humidity": humidity}, index=time_index)

# Plot original data
plt.figure(figsize=(10, 5))
plt.plot(data.index, data["temperature"], "bo-", label="Original Temperature")
plt.plot(data.index, data["humidity"], "ro-", label="Original Humidity")
plt.xlabel("Time")
plt.ylabel("Values")
plt.title("Before Resampling")
plt.legend()
plt.grid()
plt.show()

# Resample every 10 minutes, taking the last recorded value
data_resampled = data.resample("10min").last()

# Plot resampled data
plt.figure(figsize=(10, 5))
plt.plot(data_resampled.index, data_resampled["temperature"], "bs-", label="Resampled Temperature")
plt.plot(data_resampled.index, data_resampled["humidity"], "rs-", label="Resampled Humidity")
plt.xlabel("Time")
plt.ylabel("Values")
plt.title("After Resampling (10 min)")
plt.legend()
plt.grid()
plt.show()



