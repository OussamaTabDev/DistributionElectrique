import pandas as pd

# Load the uploaded dataset
file_path = "fault_data.csv"
df = pd.read_csv(file_path)

# Show basic info and first few rows
df_info = df.info()
df_head = df.head()
df.describe(include='all')

