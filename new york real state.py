import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
import matplotlib.colors as mcolors

# Load NYC Real Estate Data
df = pd.read_csv('C:\\Users\\LENOVO\\Downloads\\archive(3)\\nyc-rolling-sales.csv')

# Basic Cleaning
df.columns = df.columns.str.strip().str.upper()
df['SALE PRICE'] = pd.to_numeric(df['SALE PRICE'], errors='coerce')
df['GROSS SQUARE FEET'] = pd.to_numeric(df['GROSS SQUARE FEET'], errors='coerce')

# Keep only sales with valid price
df = df[df['SALE PRICE'] > 10000]  # Remove zero or suspicious prices

# --- Simulate lat/lon by ZIP code ---
zip_latlon = {
    10001: (40.7506, -73.9970),
    11201: (40.6943, -73.9918),
    10453: (40.8564, -73.9120),
    10301: (40.6318, -74.0944),
    11368: (40.7498, -73.8619),
    10002: (40.7170, -73.9870),
    11211: (40.7098, -73.9574),
    10462: (40.8415, -73.8567),
    11385: (40.7037, -73.8886),
}

# Map lat/lon
def find_lat(zip_code):
    return zip_latlon.get(zip_code, (None, None))[0]

def find_lon(zip_code):
    return zip_latlon.get(zip_code, (None, None))[1]

df['LAT'] = df['ZIP CODE'].apply(find_lat)
df['LON'] = df['ZIP CODE'].apply(find_lon)

# Drop missing lat/lon
df = df.dropna(subset=['LAT', 'LON'])

# Work with a sample
data = df.sample(300).copy()

# Normalize sale prices
scaler = MinMaxScaler()
data['Price_Norm'] = scaler.fit_transform(data[['SALE PRICE']])
# Log-transform SALE PRICE
data['SalePrice_log'] = np.log(data['SALE PRICE'])
mu, sigma = norm.fit(data['SalePrice_log'])

# Plot histogram
plt.figure(figsize=(10,6))
sns.histplot(data['SalePrice_log'], kde=True, stat='density', color='skyblue')
x = np.linspace(min(data['SalePrice_log']), max(data['SalePrice_log']), 100)
plt.plot(x, norm.pdf(x, mu, sigma), 'r-', lw=2, label=f'Fit: mu={mu:.2f}, sigma={sigma:.2f}')
plt.title('Distribution of log(Sale Price)')
plt.xlabel('log(Sale Price)')
plt.ylabel('Density')
plt.legend()
plt.grid()
plt.show()
# Define MGF and CF
mgf = lambda t: np.exp(mu*t + 0.5*(sigma**2)*(t**2))
cf = lambda t: np.exp(1j*mu*t - 0.5*(sigma**2)*(t**2))

# Plot MGF and CF
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
m = folium.Map(location=[40.7128, -74.0060], zoom_start=11)

for _, row in data.iterrows():
    folium.CircleMarker(
        location=(row['LAT'], row['LON']),
        radius=5,
        color=mcolors.to_hex(plt.cm.viridis(row['Price_Norm'])),
        fill=True,
        fill_opacity=0.7,
        popup=f"${row['SALE PRICE']:,.0f}"
    ).add_to(m)

# Save map
m.save("C:\\Users\\LENOVO\\Downloads\\archive(3)\\nyc-rolling-sales.html")
