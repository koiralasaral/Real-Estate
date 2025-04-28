import pandas as pd
import folium
from folium.plugins import MarkerCluster
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# Load the data
df = pd.read_csv("C:\\Users\\LENOVO\\Downloads\\archive(3)\\nyc-rolling-sales.csv")

# Basic cleaning
df.columns = df.columns.str.strip().str.upper()
df['SALE PRICE'] = pd.to_numeric(df['SALE PRICE'], errors='coerce')
df['GROSS SQUARE FEET'] = pd.to_numeric(df['GROSS SQUARE FEET'], errors='coerce')
df['LAND SQUARE FEET'] = pd.to_numeric(df['LAND SQUARE FEET'], errors='coerce')
df['YEAR BUILT'] = pd.to_numeric(df['YEAR BUILT'], errors='coerce')

# Remove entries with missing or 0 sale price
df = df[df['SALE PRICE'] > 0]

# Define borough coordinates (rough centroids)
borough_locations = {
    1: (40.7831, -73.9712),   # Manhattan
    2: (40.8448, -73.8648),   # Bronx
    3: (40.6782, -73.9442),   # Brooklyn
    4: (40.7282, -73.7949),   # Queens
    5: (40.5795, -74.1502)    # Staten Island
}

# Create Folium Map
m = folium.Map(location=[40.7128, -74.0060], zoom_start=11)
marker_cluster = MarkerCluster().add_to(m)

# Function to color by price
def get_color(price):
    if price < 500000:
        return 'green'
    elif price < 2000000:
        return 'orange'
    else:
        return 'red'

# Sample 500 points for faster plotting
for idx, row in df.sample(500).iterrows():
    borough = row['BOROUGH']
    if borough in borough_locations:
        lat, lon = borough_locations[borough]
        folium.CircleMarker(
            location=[lat + (0.01 * (0.5 - np.random.rand())), 
                      lon + (0.01 * (0.5 - np.random.rand()))],  # Add slight jitter
            radius=5,
            color=get_color(row['SALE PRICE']),
            fill=True,
            fill_color=get_color(row['SALE PRICE']),
            fill_opacity=0.7,
            popup=f"${row['SALE PRICE']:,.0f}"
        ).add_to(marker_cluster)

# Save the map
m.save('C:\\Users\\LENOVO\\Downloads\\archive(3)\\nyc-rolling-sales-ex.html')


# --- Expanded Heatmap ---

# Select more variables
heatmap_vars = ['SALE PRICE', 'GROSS SQUARE FEET', 'LAND SQUARE FEET', 'YEAR BUILT']
corr = df[heatmap_vars].corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='YlGnBu', fmt=".2f")
plt.title('Correlation Heatmap: Real Estate Features')
plt.show()


# --- Bonus Plot: Price vs Size ---

plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='GROSS SQUARE FEET', y='SALE PRICE', alpha=0.5)
plt.title('Sale Price vs Gross Square Feet')
plt.xlabel('Gross Square Feet')
plt.ylabel('Sale Price ($)')
plt.xlim(0, 10000)
plt.ylim(0, 10000000)
plt.show()
