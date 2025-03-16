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
# Define a cache list to append data and
# Ocassionally clear cache 

cache = []

def on_message(client, userdata, message):
 	# Combine timestamp and payload
    data = f"{message.timestamp},{message.payload}"
    # Append data to cache
    cache.append(data)
    # Check cache length
    if len(cache)> MAX_CACHE:
        with Path("energy.txt").open("a") as f:
            # Save to file
            f.writelines(cache)
        # reset cache
        cache.clear()
--------------------------------------------------------------------------------------------------------
# Convert the timestamp
df["ts"] = pd.to_datetime(df["ts"], unit="ms")

# Print datatypes and first observations
print(df.dtypes)
print(df.head())
-------------------------------------------------------------------------------------------------------
# Replace the timestamp with the parsed timestamp
df['ts'] = pd.to_datetime(df["ts"], unit="ms")
print(df.head())

------------------------------------------------------------------------------------------------------
# Replace the timestamp with the parsed timestamp
df['ts'] = pd.to_datetime(df["ts"], unit="ms")
print(df.head())
help(pd.pivot_table)

# Pivot the DataFrame
df2 = pd.pivot_table(df, values="val", index="ts", columns ="device")
print(df2.head())
-----------------------------------------------------------------------------------------------------
# Resampling to deal to gain none consistent data intervals
# Drops missing values in each sample
# Replace the timestamp with the parsed timestamp

df['ts'] = pd.to_datetime(df["ts"], unit="ms")
print(df.head())

# Pivot the DataFrame
df2 = pd.pivot_table(df, columns="device", values="val", index="ts")
print(df2.head())

# Resample DataFrame to 1min
df3 = df2.resample("1min").max().dropna()
print(df3.head())

df3.to_csv(TARGET_FILE)
------------------------------------------------------------------------------------------------
df_res= df.resample("30minutes").max()

# Get difference between values
df_diff = df_res.diff()

# Plot the DataFrame
df_diff.plot()
plt.show()

-------------------------------------------------------------------------------------------------
df_res= df.resample("30min").max()

# Get difference between values
df_diff = df_res.diff()

# Plot the DataFrame
df_diff.plot()
plt.show()

------------------------------------------------------------------------------------------------
# Resample df to 30 minutes
df_res = df.resample('30min').max()

# Get difference between values
df_diff = df_res.diff()

# Get the percent changed
df_pct = df_diff.pct_change()

# Plot the DataFrame
df_pct.plot()
plt.show()

-----------------------------------------------------------------------------------------------

This Django API now:

✅ Consumes MQTT messages and stores them in a database
✅ Provides REST endpoints for sensor data retrieval
✅ Alerts when temperature/smoke thresholds exceed limits
--------------------------------------------------------------------------------------------------
# Rename the columns
temperature.columns = ["temperature"]
humidity.columns = ["humidity"]
windspeed.columns =["windspeed"]

# Create list of DataFrames
df_list = [temperature, humidity, windspeed]

# Concatenate files
environment = pd.concat(df_list, axis=1)

# Print first rows of the DataFrame
print(environment.head())
-------------------------------------------------------------------------------------------------
# Combine the DataFrames
environ_traffic = pd.concat([environ, traffic], axis=1)

# Print first 5 rows
print(environ_traffic.head())
-------------------------------------------------------------------------------------------------
# Combine the DataFrames
environ_traffic = pd.concat([environ, traffic], axis=1)

# Print first 5 rows
print(environ_traffic.head())

# Create agg logic
agg_dict = {"temperature": "max", "humidity": "max", "sunshine": "sum", 
            "light_veh": "sum", "heavy_veh": "sum",
            }
-------------------------------------------------------------------------------------------------
# Combine the DataFrames
environ_traffic = pd.concat([environ, traffic], axis=1)

# Print first 5 rows
print(environ_traffic.head())

# Create agg logic
agg_dict = {"temperature": "max", "humidity": "max", "sunshine": "sum", 
            "light_veh": "sum", "heavy_veh": "sum",
            }

# Resample the DataFrame 
environ_traffic_resampled = environ_traffic.resample("1h").agg(agg_dict)
print(environ_traffic_resampled.head())
--------------------------------------------------------------------------------------------
# Combine the DataFrames
environ_traffic = pd.concat([environ, traffic], axis=1)

# Print first 5 rows
print(environ_traffic.head())

# Create agg logic
agg_dict = {"temperature": "max", "humidity": "max", "sunshine": "sum", 
            "light_veh": "sum", "heavy_veh": "sum",
            }

# Resample the DataFrame 
environ_traffic_resampled = environ_traffic.resample("1h").agg(agg_dict)
print(environ_traffic_resampled.head())

----------------------------------------------------------------------------------------------
# Calculate correlation
corr=data.corr()

# Print correlation
print(corr)
----------------------------------------------------------------------------------------------
# Calculate correlation
corr = data.corr()

# Print correlation
print(corr)

# Create a heatmap
sns.heatmap(corr, annot=True)

# Show plot
plt.show()
--------------------------------------------------------------------------------------------
# Import required modules
import seaborn as sns

# Create a pairplot
sns.pairplot(data)

# Show plot
plt.show()

-------------------------------------------------------------------------------------------
# Calculate mean
data["mean"] = data["temperature"].mean()

# Calculate upper and lower limits
data["upper_limit"] = data["mean"] + data["temperature"].std()*3
data["lower_limit"] = data["mean"] - data["temperature"].std()*3

# Plot the DataFrame
data.plot()

plt.show()
-----------------------------------------------------------------------------------------

#Summary

# Pivoting Data: You used pd.pivot_table() to transform your data, making each device a column and using timestamps as the index. This helps in easier comparison between devices.
df_pivot = df.pivot_table(index='timestamp', columns='device', values='value')

#Downsampling: To handle the sub-second differences between device events, you downsampled the data to 1-minute intervals using resample() and aggregated with max(). 
# This reduces noise and data volume.
df_resampled = df_pivot.resample('1T').max().dropna()

#Calculating Differences: To understand energy consumption over time, you calculated the difference between consecutive measurements using diff().
df_diff = df_resampled.diff(periods=1)

#Percentage Change: By applying pct_change(), you calculated the percent change between consecutive rows, allowing for better comparison between devices.
df_pct_change = df_diff.pct_change()

#These steps prepared your data for more insightful analysis and visualization.
#The goal of the next lesson is to learn how to combine and analyze IoT data from different sources with varying time intervals
--------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Plot traffic dataset before 2018-11-10
traffic[:"2018-11-10"].plot()

# Show plot
traffic.plot()
plt.show()
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Plot traffic dataset
traffic[:"2018-11-10"].plot()

# Show plot
plt.show()

# Import tsaplots
from statsmodels.graphics import tsaplots

# Plot autocorrelation
tsaplots.plot_acf(traffic["vehicles"], lags=50)

# Show the plot
plt.show()
-----------------------------------------------------------------------------------------------------------------------
# Import modules
import statsmodels.api as sm

# Perform decompositon 
res = sm.tsa.seasonal_decompose(traffic["vehicles"])

# Print the seasonal component
print(res.seasonal)

# Plot the result
res.plot()

# Show the plot
plt.show()
----------------------------------------------------------------------------------------
# Resample DataFrame to 1h
df_seas = df.resample('1h').max()

# Run seasonal decompose
decomp = sm.tsa.seasonal_decompose(df_seas)
---------------------------------------------------------------------------------------
# Resample DataFrame to 1h
df_seas = df.resample('1h').max()

# Run seasonal decompose
decomp = sm.tsa.seasonal_decompose(df_seas)

# Plot the timeseries
plt.title("Temperature")
plt.plot(df_seas["temperature"], label="temperature")

# Plot trend and seasonality
plt.plot(decomp.trend, label="trend")
plt.plot(decomp.seasonal, label="seasonal")
plt.legend()
plt.show()
--------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------
#Machine Learning Pipelines
# Define the split day
limit_day = "2018-10-27"

# Split the data
train_env = environment[:limit_day]
test_env = environment[limit_day:]
--------------------------------------------------------------------------------------------

# Define the split day
limit_day = "2018-10-27"

# Split the data
train_env = environment[:limit_day]
test_env = environment[limit_day:]

# Print start and end dates
print(show_start_end(train_env))
print(show_start_end(test_env))

# Split the data into X and y
X_train = train_env.drop("target", axis=1)
y_train = train_env["target"]
X_test = test_env.drop("target", axis=1)
y_test = test_env["target"]

-------------------------------------------------------------------------------------------
# Define the split day
limit_day = "2018-10-27"

# Split the data
train_env = environment[:limit_day]
test_env = environment[limit_day:]

# Print start and end dates
print(show_start_end(train_env))
print(show_start_end(test_env))

# Split the data into X and y
X_train = train_env.drop("target", axis=1)
y_train = train_env["target"]
X_test = test_env.drop("target", axis=1)
y_test = test_env["target"]
---------------------------------------------------------------------------------------
# Create LogisticRegression model
logreg = LogisticRegression()

# Fit the model
logreg.fit(X_train, y_train)

# Score the model
print(logreg.score(X_train, y_train))
print(logreg.score(X_test, y_test))
--------------------------------------------------------------------------------------
# Import StandardScaler
from sklearn.preprocessing import StandardScaler

# Initialize StandardScaler
sc = StandardScaler()

# Fit the scaler
sc.fit(environment)

# Print mean and variance
print(sc.mean_)
print(sc.var_)
--------------------------------------------------------------------------------------
#Scaling
# Initialize StandardScaler
sc = StandardScaler()

# Fit the scaler
sc.fit(environment)

# Transform the data
environ_scaled = sc.transform(environment)

# Convert scaled data to DataFrame
environ_scaled = pd.DataFrame(environ_scaled, 
                              columns=environment.columns, 
                              index=environment.index)
print(environ_scaled.head())
plot_unscaled_scaled(environment, environ_scaled)

--------------------------------------------------------------------------------------
# Import pipeline
from sklearn.preprocessing import StandardScaler

# Create Scaler and Regression objects
sc = StandardScaler()
logreg = LogisticRegression()
--------------------------------------------------------------------------------------
# Import pipeline
from sklearn.pipeline import Pipeline

# Create Scaler and Regression objects
sc = StandardScaler()
logreg = LogisticRegression()

# Create Pipeline
pl = Pipeline([
        ("scale", sc),
        ("logreg", logreg)
    ])
---------------------------------------------------------------------------------------
# Import pipeline
from sklearn.pipeline import Pipeline

# Create Scaler and Regression objects
sc = StandardScaler()
logreg = LogisticRegression()

# Create Pipeline
pl = Pipeline([
        ("scale", sc),
        ("logreg", logreg)
    ])

# Fit the pipeline and print predictions
pl.fit(X_train, y_train)
print(pl.predict(X_test))
---------------------------------------------------------------------------------------
# Create Pipeline
pl = Pipeline([
        ("scale", StandardScaler()),
        ("logreg", LogisticRegression())
    ])

# Fit the pipeline
pl.fit(X_train, y_train)

# Predict classes
predictions = pl.predict(X_test)

# Print results
print(predictions)
--------------------------------------------------------------------------------------
def model_subscribe(client, userdata, message):
    data = json.loads(message.payload)
    # Parse to DataFrame
    df = pd.DataFrame.from_records([data], index="timestamp", columns=cols)
    # Predict result
    category = pl.predict(df)
    if category[0] < 1:
        # Call business logic
        close_window(df, category[0])
    else:
        print("Nice Weather, nothing to do.")

# Subscribe model_subscribe to MQTT Topic
subscribe.callback(model_subscribe, topic, hostname=MQTT_HOST)
---------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------

#a Django API that retrieves and analyzes stored fire alarm data. The API will:

# ✅ Ingest data from CSV (or MQTT directly in the future)
# ✅ Expose endpoints to query sensor data
# ✅ Provide alerts based on threshold values (e.g., high smoke levels, extreme temperatures)
# ✅ Integrate with AWS S3 for storage (optional for scalability)
                       
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
------------------------------------------------------------------------------------------
#Resambling data exercise

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
----------------------------------------------------------------------------------------------------------
#Dockerise the app for deployment

FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Django project files
COPY . .

# Expose port 8000 for Django
EXPOSE 8000

# Run migrations and start server
CMD ["sh", "-c", "python manage.py migrate && python manage.py runserver 0.0.0.0:8000"]
-----------------------------------------------------------------------------------------------------------



