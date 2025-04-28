import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import pymc as pm  # Bayesian inference library
import folium
import branca.colormap as cm

############################################
# STEP A: Geocode All Neighborhoods & Update CSV
############################################

# Path to your original sales CSV; adjust as needed.
sales_csv = r"C:\\Users\\LENOVO\\Downloads\\archive(3)\\nyc-rolling-sales.csv"

# Load the sales data; assumes a "NEIGHBORHOOD" column.
data = pd.read_csv(sales_csv)

# Ensure the NEIGHBORHOOD column is uppercase and trimmed.
data['NEIGHBORHOOD'] = data['NEIGHBORHOOD'].astype(str).str.upper().str.strip()

# Create new columns for neighborhood latitude and longitude.
data['LAT_NBHD'] = None
data['LON_NBHD'] = None

# Initialize geolocator with a rate limiter.
geolocator = Nominatim(user_agent="nyc_neighborhood_locator")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

# Process every unique neighborhood.
unique_neighborhoods = data['NEIGHBORHOOD'].unique()
neighborhood_coords = {}

print("Geocoding all unique neighborhoods:")
for nbhd in unique_neighborhoods:
    query = f"{nbhd}, New York, NY"
    location = geocode(query)
    if location:
        neighborhood_coords[nbhd] = (location.latitude, location.longitude)
        print(f"  {nbhd}: {location.latitude}, {location.longitude}")
    else:
        neighborhood_coords[nbhd] = (np.nan, np.nan)
        print(f"  {nbhd}: Not found")

# Fill the LAT_NBHD and LON_NBHD columns with the geocoded values.
data['LAT_NBHD'] = data['NEIGHBORHOOD'].map(lambda nb: neighborhood_coords.get(nb, (np.nan, np.nan))[0])
data['LON_NBHD'] = data['NEIGHBORHOOD'].map(lambda nb: neighborhood_coords.get(nb, (np.nan, np.nan))[1])

# Save the updated CSV for later use.
updated_sales_csv = 'nyc-rolling-sales-with-all-neighborhood-coords.csv'
data.to_csv(updated_sales_csv, index=False)
print(f"\nUpdated CSV file saved as '{updated_sales_csv}'.")

############################################
# STEP B: Clean Data & Simulate Observed Profit
############################################

# Clean the SALE PRICE field
data['SALE PRICE'] = pd.to_numeric(
    data['SALE PRICE'].astype(str).str.replace(r'[\$,]', '', regex=True),
    errors='coerce'
)
data = data.dropna(subset=['SALE PRICE', 'NEIGHBORHOOD', 'LAT_NBHD', 'LON_NBHD'])

# For our gold standard test, we simulate an "observed profit" as follows:
#   observed_profit = SALE PRICE * true_margin  + noise
# We assume that the true_margin is 10% for all sales, but add noise.
# (noise scale is taken as 2% of the sale price)
np.random.seed(42)  # for reproducibility
data['observed_profit'] = data['SALE PRICE'] * 0.10 + np.random.normal(0, 1, len(data)) * (data['SALE PRICE'] * 0.02)

# Map neighborhood names to index numbers for hierarchical analysis.
neighborhoods = data['NEIGHBORHOOD'].unique()
n_neighborhoods = len(neighborhoods)
nbhd_to_idx = {nb: i for i, nb in enumerate(neighborhoods)}
data['nbhd_idx'] = data['NEIGHBORHOOD'].map(nbhd_to_idx)

############################################
# STEP C: Hierarchical Bayesian Inference with PyMC3
############################################

print("\nRunning hierarchical Bayesian inference...")

with pm.Model() as hierarchical_model:
    # Global (hyper-)priors for the profit margin.
    mu_margin = pm.Normal("mu_margin", mu=0.10, sigma=0.05)  # overall mean margin
    sigma_margin = pm.HalfNormal("sigma_margin", sigma=0.05)
    
    # For each neighborhood j, the margin is drawn from a Normal around mu_margin.
    margin = pm.Normal("margin", mu=mu_margin, sigma=sigma_margin, shape=n_neighborhoods)
    
    # Noise in the observed profit measurement.
    sigma_obs = pm.HalfNormal("sigma_obs", sigma=1e5)
    
    # For each sale, the model assumes:
    #      observed_profit = SALE PRICE * margin[nbhd_idx]  + noise
    mu_profit = data['SALE PRICE'].values * margin[data['nbhd_idx'].values]
    observed = pm.Normal("observed", mu=mu_profit, sigma=sigma_obs, 
                         observed=data['observed_profit'].values)
    
    trace = pm.sample(2000, tune=1000, target_accept=0.95, return_inferencedata=False)

# Calculate posterior summaries for each neighborhood's margin.
posterior_margin_mean = trace["margin"].mean(axis=0)
posterior_margin_hpd = pm.stats.hpd(trace["margin"], hdi_prob=0.90)

##############################
# STEP D: Matplotlib Visualization
##############################

import matplotlib.ticker as mtick

plt.figure(figsize=(12, 6))
bars = plt.bar(range(n_neighborhoods), posterior_margin_mean, yerr=[posterior_margin_mean - posterior_margin_hpd[:,0],
                                                                      posterior_margin_hpd[:,1] - posterior_margin_mean],
                capsize=5, color='skyblue', alpha=0.8)
plt.xticks(range(n_neighborhoods), neighborhoods, rotation=90)
plt.ylabel("Estimated Profit Margin")
plt.title("Posterior Estimated Profit Margin by Neighborhood")
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.tight_layout()
plt.show()

##############################
# STEP E: Folium Map Visualization
##############################

# Create a base Folium map centered around New York City.
m = folium.Map(location=[40.7128, -74.0060], zoom_start=11)

# Create a linear colormap (from blue to red) according to estimated margin.
min_margin = posterior_margin_mean.min()
max_margin = posterior_margin_mean.max()
colormap = cm.LinearColormap(colors=['blue', 'green', 'yellow', 'red'],
                              vmin=min_margin, vmax=max_margin,
                              caption='Estimated Profit Margin')

# For each neighborhood, add a marker with popup text showing the estimated margin.
for nbhd in neighborhoods:
    # Get the first row in data for that neighborhood (to retrieve the coordinates)
    nbhd_data = data[data['NEIGHBORHOOD'] == nbhd].iloc[0]
    lat = nbhd_data['LAT_NBHD']
    lon = nbhd_data['LON_NBHD']
    idx = nbhd_to_idx[nbhd]
    est_margin = posterior_margin_mean[idx]
    popup_text = f"<b>{nbhd}</b><br>Estimated Profit Margin: {est_margin:.2%}"
    folium.CircleMarker(location=[lat, lon],
                        radius=7,
                        color=colormap(est_margin),
                        fill=True,
                        fill_color=colormap(est_margin),
                        fill_opacity=0.7,
                        popup=popup_text).add_to(m)

# Add the colormap to the map.
colormap.add_to(m)

# Save the map to an HTML file and display the file path.
folium_map_file = "nyc_profit_margin_map.html"
m.save(folium_map_file)
print(f"\nFolium map saved as '{folium_map_file}'.")
