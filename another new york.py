# --- 1. Import Libraries ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
from scipy import stats
from scipy.stats import norm
from scipy.interpolate import griddata
import matplotlib.colors as mcolors
from sklearn.preprocessing import MinMaxScaler

# --- 2. Load and Clean NYC Data ---
df = pd.read_csv("C:\\Users\\LENOVO\\Downloads\\archive(3)\\nyc-rolling-sales.csv")


df.columns = df.columns.str.strip().str.upper()
df['SALE PRICE'] = pd.to_numeric(df['SALE PRICE'], errors='coerce')
df['GROSS SQUARE FEET'] = pd.to_numeric(df['GROSS SQUARE FEET'], errors='coerce')
df['LAND SQUARE FEET'] = pd.to_numeric(df['LAND SQUARE FEET'], errors='coerce')
df['YEAR BUILT'] = pd.to_numeric(df['YEAR BUILT'], errors='coerce')

# Filter valid sales
df = df[df['SALE PRICE'] > 0]

# Approximate Lat/Lon from ZIP Code
zip_latlon = {
    10001: (40.7506, -73.9970),
    10002: (40.7170, -73.9870),
    11201: (40.6943, -73.9918),
    11211: (40.7098, -73.9574),
    10453: (40.8564, -73.9120),
    10462: (40.8415, -73.8567),
    10301: (40.6318, -74.0944),
    11368: (40.7498, -73.8619),
    11385: (40.7037, -73.8886),
}

def find_lat(zip_code):
    return zip_latlon.get(zip_code, (None, None))[0]

def find_lon(zip_code):
    return zip_latlon.get(zip_code, (None, None))[1]

df['LATITUDE'] = df['ZIP CODE'].apply(find_lat)
df['LONGITUDE'] = df['ZIP CODE'].apply(find_lon)

df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])

# Normalize SALE PRICE for color mapping
scaler = MinMaxScaler()
df['PRICE_NORM'] = scaler.fit_transform(df[['SALE PRICE']])

# --- 3. Probability Distribution (log-normal) ---
df['SALE PRICE LOG'] = np.log(df['SALE PRICE'])
mu, sigma = norm.fit(df['SALE PRICE LOG'])

plt.figure(figsize=(10,6))
sns.histplot(df['SALE PRICE LOG'], kde=True, stat='density', color='skyblue')
x = np.linspace(min(df['SALE PRICE LOG']), max(df['SALE PRICE LOG']), 100)
plt.plot(x, norm.pdf(x, mu, sigma), 'r-', lw=2, label=f'Fit: mu={mu:.2f}, sigma={sigma:.2f}')
plt.title('Distribution of log(Sale Price)')
plt.xlabel('log(Sale Price)')
plt.ylabel('Density')
plt.legend()
plt.grid()
plt.show()

# --- 4. Folium Map ---
m = folium.Map(location=[40.7128, -74.0060], zoom_start=11)
marker_cluster = MarkerCluster().add_to(m)

for _, row in df.iterrows():
    folium.CircleMarker(
        location=(row['LATITUDE'], row['LONGITUDE']),
        radius=5,
        color=mcolors.to_hex(plt.cm.viridis(row['PRICE_NORM'])),
        fill=True,
        fill_color=mcolors.to_hex(plt.cm.viridis(row['PRICE_NORM'])),
        fill_opacity=0.7,
        popup=f"${row['SALE PRICE']:,.0f}"
    ).add_to(marker_cluster)

# Save the Folium Map
m.save('C:\\Users\\LENOVO\\Downloads\\archive(3)\\nyc-rolling-sales_map.html')

# --- 5. Contour and Gradient Analysis ---
grid_x, grid_y = np.mgrid[
    min(df['LONGITUDE']):max(df['LONGITUDE']):200j,
    min(df['LATITUDE']):max(df['LATITUDE']):200j
]
points = np.vstack((df['LONGITUDE'], df['LATITUDE'])).T
values = df['SALE PRICE']

grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')

# Gradients
grad_y, grad_x = np.gradient(grid_z)

plt.figure(figsize=(10,8))
cp = plt.contourf(grid_x, grid_y, grid_z, cmap='viridis', levels=20)
plt.colorbar(cp)
plt.quiver(grid_x, grid_y, grad_x, grad_y, color='white', scale=1e8)
plt.title('Contour and Gradient of Sale Prices in NYC')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid()
plt.show()

# --- 6. Monte Carlo Simulation (Buy/Sell Profits) ---
num_simulations = 1000
price_fluctuation = 0.1  # +/-10% price fluctuation

simulation_results = []

for i, row in df.iterrows():
    lat, lon = row['LATITUDE'], row['LONGITUDE']
    actual_price = row['SALE PRICE']
    
    simulated_buy = np.random.normal(actual_price, price_fluctuation * actual_price, num_simulations)
    simulated_sell = np.random.normal(actual_price, price_fluctuation * actual_price, num_simulations)
    
    profits = simulated_sell - simulated_buy
    
    simulation_results.append({
        'Latitude': lat,
        'Longitude': lon,
        'Actual_Price': actual_price,
        'Simulated_Profit_Mean': np.mean(profits),
        'Simulated_Profit_Std': np.std(profits)
    })

sim_data = pd.DataFrame(simulation_results)

# --- 7. Heatmap of Simulated Profits ---
sim_data['Latitude_round'] = sim_data['Latitude'].round(3)
sim_data['Longitude_round'] = sim_data['Longitude'].round(3)

heatmap_data = sim_data.groupby(['Latitude_round', 'Longitude_round'])['Simulated_Profit_Mean'].mean().unstack()

plt.figure(figsize=(10,8))
sns.heatmap(heatmap_data, cmap='coolwarm', annot=False)
plt.title('Heatmap of Simulated Profits - NYC Real Estate')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
