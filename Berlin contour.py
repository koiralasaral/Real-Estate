# Berlin Real Estate Probability Estimation

## 1. Import Libraries


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler


## 2. Get Real Data (Berlin Real Estate)


# Load real data from a public dataset (e.g., Kaggle, OpenStreetMap, or manual data)
# Here we simulate because of offline environment
np.random.seed(42)
n = 200
area = np.random.uniform(30, 150, n)  # Area in square meters
cost_per_m2 = np.random.uniform(3000, 8000, n)  # Cost per m^2 in EUR
cost = area * cost_per_m2

# Simulate coordinates around Berlin
latitudes = np.random.uniform(52.48, 52.55, n)
longitudes = np.random.uniform(13.35, 13.45, n)

# Create DataFrame
data = pd.DataFrame({
    'Area_m2': area,
    'Cost_per_m2': cost_per_m2,
    'Total_Cost': cost,
    'Latitude': latitudes,
    'Longitude': longitudes
})

# Normalize cost for color mapping
scaler = MinMaxScaler()
data['Cost_Norm'] = scaler.fit_transform(data[['Total_Cost']])


## 3. Probability Distribution


# Assume Total_Cost follows a normal distribution
data['Total_Cost_log'] = np.log(data['Total_Cost'])
mu, sigma = norm.fit(data['Total_Cost_log'])

# Plot histogram and fitted curve
plt.figure(figsize=(10,6))
sns.histplot(data['Total_Cost_log'], kde=True, stat='density', color='skyblue')
x = np.linspace(min(data['Total_Cost_log']), max(data['Total_Cost_log']), 100)
plt.plot(x, norm.pdf(x, mu, sigma), 'r-', lw=2, label=f'Fit: mu={mu:.2f}, sigma={sigma:.2f}')
plt.title('Distribution of log(Total Cost)')
plt.xlabel('log(Total Cost)')
plt.ylabel('Density')
plt.legend()
plt.grid()
plt.show()


## 4. Probability Generating Function (PGF), Moment Generating Function (MGF), Characteristic Function (CF)


# Define PGF, MGF, CF for normal (lognormal in our model)

# Moment Generating Function (MGF) for Normal
mgf = lambda t: np.exp(mu*t + 0.5*(sigma**2)*(t**2))

# Characteristic Function (CF)
cf = lambda t: np.exp(1j*mu*t - 0.5*(sigma**2)*(t**2))

# Plot MGF
t_vals = np.linspace(-2, 2, 400)
mgf_vals = mgf(t_vals)
cf_vals = cf(t_vals)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(t_vals, mgf_vals.real)
plt.title('Moment Generating Function (MGF)')
plt.xlabel('t')
plt.ylabel('MGF(t)')
plt.grid()

plt.subplot(1,2,2)
plt.plot(t_vals, cf_vals.real, label='Real Part')
plt.plot(t_vals, cf_vals.imag, label='Imag Part', linestyle='--')
plt.title('Characteristic Function (CF)')
plt.xlabel('t')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()



'''- The distribution of Total Cost (log-transformed) approximates a **Normal Distribution**.

---

## 5. Scatter Plot on Map (Folium)
'''
m = folium.Map(location=[52.52, 13.40], zoom_start=12)

for _, row in data.iterrows():
    folium.CircleMarker(
        location=(row['Latitude'], row['Longitude']),
        radius=5,
        color=plt.cm.viridis(row['Cost_Norm']),
        fill=True,
        fill_opacity=0.7,
        popup=f"€{row['Total_Cost']:.0f}"
    ).add_to(m)

m
'''. Contour Lines & Gradient (Differential Calculus)

'''
from scipy.interpolate import griddata

# Create grid
grid_x, grid_y = np.mgrid[min(data['Longitude']):max(data['Longitude']):200j,
                           min(data['Latitude']):max(data['Latitude']):200j]
points = np.vstack((data['Longitude'], data['Latitude'])).T
values = data['Total_Cost']

# Interpolate
grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')

# Compute gradients
grad_y, grad_x = np.gradient(grid_z)

# Plot
plt.figure(figsize=(10,8))
cp = plt.contourf(grid_x, grid_y, grid_z, cmap='viridis', levels=20)
plt.colorbar(cp)
plt.quiver(grid_x, grid_y, grad_x, grad_y, color='white', scale=1e8)
plt.title('Contour and Gradient (Differential Calculus) of Total Cost')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid()
plt.show()
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Monte Carlo Simulation Parameters ---
num_simulations = 1000
price_fluctuation = 0.1  # +/-10% fluctuation in buy/sell prices

# Store results
simulation_results = []

# Run simulations for each coordinate
for i, row in data.iterrows():
    lat, lon = row['Latitude'], row['Longitude']
    actual_cost = row['Total_Cost']
    
    # Simulate random fluctuations in buy and sell prices
    simulated_buy_prices = np.random.normal(actual_cost, price_fluctuation * actual_cost, num_simulations)
    simulated_sell_prices = np.random.normal(actual_cost, price_fluctuation * actual_cost, num_simulations)
    
    # Calculate profit/loss
    profits = simulated_sell_prices - simulated_buy_prices
    
    # Append results
    simulation_results.append({
        'Latitude': lat,
        'Longitude': lon,
        'Actual_Cost': actual_cost,
        'Simulated_Profit_Mean': np.mean(profits),
        'Simulated_Profit_Std': np.std(profits)
    })

# Convert to DataFrame
sim_data = pd.DataFrame(simulation_results)

# Print intermediate values (for 2 coordinates as an example)
print(sim_data.head(2))

# --- Create Seaborn Heatmap of Simulated Profits ---
# Aggregate profits geographically (grid-based)
heatmap_data = sim_data.groupby(['Latitude', 'Longitude'])['Simulated_Profit_Mean'].mean().unstack()

plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, cmap='coolwarm', annot=False)
plt.title('Heatmap of Simulated Profits')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# --- Compare Simulation to Actual Data ---
# Print original data and compare with simulated results
comparison = data[['Latitude', 'Longitude', 'Total_Cost']].merge(sim_data, on=['Latitude', 'Longitude'])
print(comparison.head())
