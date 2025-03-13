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

