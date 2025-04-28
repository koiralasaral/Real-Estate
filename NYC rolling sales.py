# -------------------------------
# 1. Import Libraries and Set-up
# -------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import griddata

# For reproducibility:
np.random.seed(42)

# -------------------------------
# 2. Load and Clean the NYC Dataset
# -------------------------------
# Assume you have downloaded "NY-House-Dataset.csv" from Kaggle.
try:
    data = pd.read_csv("C:\\Users\\LENOVO\\Downloads\\archive(3)\\nyc-rolling-sales.csv")
except FileNotFoundError:
    raise FileNotFoundError("Ensure 'nyc-rolling-sales.csv' is in your working directory.")

# Inspect a few rows and columns:
print("Original dataset columns:", data.columns.tolist())
print(data.head())

# For this example, we focus on columns: PRICE, LATITUDE, LONGITUDE.
# Convert PRICE to numeric (if it isn’t already) and drop rows with missing or non-positive prices.
data['SALE PRICE'] = pd.to_numeric(data['SALE PRICE'], errors='coerce')
data = data.dropna(subset=['SALE PRICE', 'LATITUDE', 'LONGITUDE'])
data = data[data['SALE PRICE'] > 0]

# You might also filter by state if necessary. For NYC properties, ensure STATE column contains 'NY'
if 'STATE' in data.columns:
    data = data[data['STATE'].str.contains("NY", na=False)]

# Optionally, sample a subset for simulation demonstration (e.g., first 200 listings)
data = data.head(200)

print("\nCleaned data summary:")
print(data[['PRICE', 'LATITUDE', 'LONGITUDE']].describe())

# -------------------------------
# 3. Monte Carlo Simulation for Buy & Sale Prices
# -------------------------------
# We assume that for each property the "actual sale price" is our baseline.
# We simulate buy and sell price fluctuations at ±10% (can be adjusted) around the actual price.
num_simulations = 1000
price_fluctuation = 0.1  # 10% fluctuation

simulation_results = []

# Loop through each property (using DataFrame.iterrows for demonstration)
for index, row in data.iterrows():
    base_price = row['SALE PRICE']
    lat = row['LATITUDE']
    lon = row['LONGITUDE']

    # Generate simulated buy and sell prices using a normal distribution around base_price.
    simulated_buy_prices = np.random.normal(loc=base_price, scale=price_fluctuation * base_price, size=num_simulations)
    simulated_sell_prices = np.random.normal(loc=base_price, scale=price_fluctuation * base_price, size=num_simulations)
    
    # Calculate profit/loss from simulated buy and sale.
    profits = simulated_sell_prices - simulated_buy_prices
    
    simulation_results.append({
        'LATITUDE': lat,
        'LONGITUDE': lon,
        'Actual_Price': base_price,
        'Simulated_Profit_Mean': np.mean(profits),
        'Simulated_Profit_Std': np.std(profits)
    })

# Convert simulation results into a DataFrame.
sim_data = pd.DataFrame(simulation_results)

# Print intermediate simulation results for the first 2 properties.
print("\nIntermediate simulation results (first 2 rows):")
print(sim_data.head(2))

# -------------------------------
# 4. Create a Seaborn Heatmap of Simulated Profits
# -------------------------------
# For a heatmap we need to aggregate the simulated profit means spatially.
# One approach: pivot the simulation results by rounding lat/lon to a grid.
sim_data['lat_round'] = sim_data['LATITUDE'].round(3)
sim_data['lon_round'] = sim_data['LONGITUDE'].round(3)

pivot_table = sim_data.pivot_table(index='lat_round', columns='lon_round', values='Simulated_Profit_Mean', aggfunc='mean')

plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table, cmap='coolwarm', cbar_kws={'label': 'Mean Profit'})
plt.title('Seaborn Heatmap of Simulated Profits (NYC)')
plt.xlabel('Longitude (rounded)')
plt.ylabel('Latitude (rounded)')
plt.show()

# -------------------------------
# 5. Compare Simulation with Actual Sale Prices
# -------------------------------
# We join simulation results back with the original data on LATITUDE and LONGITUDE.
comparison = data[['LATITUDE', 'LONGITUDE', 'PRICE']].merge(sim_data, on=['LATITUDE', 'LONGITUDE'])
print("\nComparison of Actual Price vs. Simulated Profit (first 5 rows):")
print(comparison[['LATITUDE', 'LONGITUDE', 'PRICE', 'Simulated_Profit_Mean', 'Simulated_Profit_Std']].head())

# Optionally, visualize this comparison on a Folium map.
# Normalize simulated profit for color mapping.
scaler = MinMaxScaler()
comparison['Profit_Norm'] = scaler.fit_transform(comparison[['Simulated_Profit_Mean']])

# Create a Folium map centered on NYC (approximate coordinates).
map_nyc = folium.Map(location=[40.75, -73.97], zoom_start=12)

for _, row in comparison.iterrows():
    # Use a color gradient based on normalized profit (scale between blue and red).
    color = plt.cm.coolwarm(row['Profit_Norm'])
    # Convert the RGBA from matplotlib colormap to a hex string.
    color_hex = matplotlib.colors.rgb2hex(color)
    
    folium.CircleMarker(
        location=(row['LATITUDE'], row['LONGITUDE']),
        radius=5,
        color=color_hex,
        fill=True,
        fill_opacity=0.7,
        popup=f"Actual Price: ${row['PRICE']:.0f}\nSimulated Profit: {row['Simulated_Profit_Mean']:.2f}"
    ).add_to(map_nyc)

# Display the map in a Jupyter Notebook or save it to file.
map_nyc.save("NYC_Real_Estate_Simulation_Map.html")
print("\nFolium map saved as NYC_Real_Estate_Simulation_Map.html")