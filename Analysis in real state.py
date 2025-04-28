import pandas as pd
import time
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import numpy as np


# Load your NYC sales data (expects a 'NEIGHBORHOOD' column)
data = pd.read_csv("C:\\Users\\LENOVO\\Downloads\\archive(3)\\nyc-rolling-sales.csv")

# Normalize the 'NEIGHBORHOOD' column: uppercase and strip extra spaces.
data['NEIGHBORHOOD'] = data['NEIGHBORHOOD'].str.upper().str.strip()

# Get a list of all unique neighborhoods in your CSV
unique_neighborhoods = data['NEIGHBORHOOD'].unique()

# Initialize the geolocator and a rate limiter (1 second delay between calls)
geolocator = Nominatim(user_agent="nyc_neighborhood_locator")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

# Dictionary to store neighborhood coordinates
neighborhood_coords = {}

print("Fetching geocode data for neighborhoods...")
for nbhd in unique_neighborhoods:
    # Construct query string for geocoding: adding "New York, NY" helps disambiguate.
    query = f"{nbhd}, New York, NY"
    location = geocode(query)
    if location:
        neighborhood_coords[nbhd] = (location.latitude, location.longitude)
        print(f"{nbhd}: {location.latitude}, {location.longitude}")
    else:
        neighborhood_coords[nbhd] = (None, None)
        print(f"{nbhd}: Not found")

# Optionally, review the collected neighborhood coordinates
print("\nAll neighborhood coordinates:")
for nbhd, coords in neighborhood_coords.items():
    print(f"{nbhd}: {coords}")

# Map the dictionary values back to the DataFrame's new columns
data['LAT_NBHD'] = data['NEIGHBORHOOD'].apply(lambda n: neighborhood_coords.get(n, (None, None))[0])
data['LON_NBHD'] = data['NEIGHBORHOOD'].apply(lambda n: neighborhood_coords.get(n, (None, None))[1])

# Save the updated CSV file with comprehensive neighborhood latitude and longitude
output_filename = 'nyc-rolling-sales-with-all-neighborhood-coords.csv'
data.to_csv(output_filename, index=False)
print(f"\nUpdated CSV file saved as '{output_filename}'.")


# Load your simulated profit dataset
sim_data = pd.read_csv("C:\\Users\\LENOVO\\Downloads\\archive(3)\\nyc-rolling-sales.csv")

# Create evenly spaced grid coordinates based on the data range
grid_lon = np.linspace(sim_data['LON_NBHD'].min(), sim_data['LON_NBHD'].max(), 100)
grid_lat = np.linspace(sim_data['LAT_NBHD'].min(), sim_data['LAT_NBHD'].max(), 100)
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

# Interpolate the profit across the grid using cubic interpolation
grid_profit = griddata(
    (sim_data['LON_NBHD'], sim_data['LAT_NBHD']),
    sim_data['Simulated_Profit_Mean'],
    (grid_lon, grid_lat),
    method='cubic'
)

# Plot the fitted profit surface
plt.figure(figsize=(10, 8))
plt.contourf(grid_lon, grid_lat, grid_profit, cmap='coolwarm')
plt.colorbar(label='Simulated Mean Profit')
plt.title('Simulated Profit Surface across NYC')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Load the updated sales data with neighborhood coordinates (use original latitude/longitude columns for spatial analysis)
data = pd.read_csv('"C:\\Users\\LENOVO\\Downloads\\archive(3)\\nyc-rolling-sales-with-coords.csv')

# Convert the 'SALE PRICE' to numeric.
# If your sale prices include currency symbols or commas, remove them:
data['SALE PRICE'] = pd.to_numeric(data['SALE PRICE'].str.replace('[\$,]', '', regex=True), errors='coerce')

# Drop rows with missing sale price or coordinates and filter out zero sale prices
data = data.dropna(subset=['SALE PRICE', 'LAT_NBMD', 'LON_NBHD'])
data = data[data['SALE PRICE'] > 0]

# Build a grid for interpolation across NYC
grid_lon = np.linspace(data['LON_NBHD'].min(), data['LONGITUDE'].max(), 100)
grid_lat = np.linspace(data['LAT_NBHD'].min(), data['LAT_NBHD'].max(), 100)
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

# Interpolate the sale prices onto the grid
grid_price = griddata(
    (data['LONGITUDE'], data['LATITUDE']),
    data['SALE PRICE'],
    (grid_lon, grid_lat),
    method='linear'
)

# Calculate the gradients (first-order differences) in both spatial directions.
# The second and third arguments tell np.gradient the distance between grid points.
unique_lat_vals = np.unique(grid_lat)
unique_lon_vals = np.unique(grid_lon)
dlat = unique_lat_vals[1] - unique_lat_vals[0] if len(unique_lat_vals) > 1 else 1.0
dlon = unique_lon_vals[1] - unique_lon_vals[0] if len(unique_lon_vals) > 1 else 1.0

grad_lat, grad_lon = np.gradient(grid_price, dlat, dlon)

# The magnitude of the gradient gives the rate of change of prices (how steep the change is)
grad_magnitude = np.sqrt(grad_lat**2 + grad_lon**2)

# Plot the gradient magnitude
plt.figure(figsize=(10, 8))
plt.contourf(grid_lon, grid_lat, grad_magnitude, levels=50, cmap='viridis')
plt.colorbar(label='Price Gradient Magnitude')
plt.title('Geographic Gradient of Sale Prices')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
