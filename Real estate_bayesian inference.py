import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import pymc as pm  # Make sure PyMC3 is installed

############################################
# SETTINGS & INPUTS
############################################

# Specify the selected neighborhood (change as needed)
selected_neighborhood = "FLUSHING"

# CSV filenames (adjust the paths if needed)
sales_csv = r"C:\Users\LENOVO\Downloads\archive(3)\nyc-rolling-sales.csv"  # original sales data file
simulated_csv = "nyc-rolling-sales-with-selected-neighborhood-coords.csv"  # file to use for simulation

############################################
# STEP 1: Update Sales CSV with Geopy for Only the Selected Neighborhood
############################################

# Load the sales CSV file (expects a 'NEIGHBORHOOD' column)
data = pd.read_csv(sales_csv)

# Ensure the NEIGHBORHOOD column is uppercase and stripped of extra spaces
data['NEIGHBORHOOD'] = data['NEIGHBORHOOD'].astype(str).str.upper().str.strip()

# Create new columns for neighborhood latitude and longitude
data['LAT_NBHD'] = None
data['LON_NBHD'] = None

# Define a mask for rows matching the selected neighborhood
mask_selected = data['NEIGHBORHOOD'] == selected_neighborhood.upper()

# Use geopy to get coordinates for the selected neighborhood
if mask_selected.sum() > 0:
    print(f"Selected neighborhood '{selected_neighborhood}' found in the data. Fetching its coordinates using geopy...")
    geolocator = Nominatim(user_agent="nyc_neighborhood_locator")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    # Construct query string (adding "New York, NY" to help disambiguate)
    query = f"{selected_neighborhood}, New York, NY"
    location = geocode(query)

    if location:
        lat, lon = location.latitude, location.longitude
        print(f"Geocoded '{selected_neighborhood}': {lat}, {lon}")
        data.loc[mask_selected, 'LAT_NBHD'] = lat
        data.loc[mask_selected, 'LON_NBHD'] = lon
    else:
        print(f"Could not geocode the selected neighborhood: '{selected_neighborhood}'.")
else:
    print(f"Neighborhood '{selected_neighborhood}' not found in the sales data.")

# Save the updated CSV file for further use
updated_sales_csv = 'nyc-rolling-sales-with-selected-neighborhood-coords.csv'
data.to_csv(updated_sales_csv, index=False)
print(f"Updated CSV file saved as '{updated_sales_csv}'.\n")

############################################
# PROGRAM 1: Fit a Profit Surface from Simulated Data using MCMC-based Profit Simulation
############################################

# Load simulated profit data from the updated CSV file.
sim_data = pd.read_csv(simulated_csv)

# Generate simulated profit using an MCMC method if not already present.
if 'Simulated_Profit_Mean' not in sim_data.columns:
    if 'SALE PRICE' in sim_data.columns:
        # Clean the SALE PRICE column: remove currency symbols and commas (using a raw string for the regex)
        sim_data['SALE PRICE'] = pd.to_numeric(
            sim_data['SALE PRICE'].astype(str).str.replace(r'[\$,]', '', regex=True),
            errors='coerce'
        )

        # --- MCMC Simulation for the profit margin ---
        # We assume the profit margin follows a normal distribution.
        # Prior: margin ~ Normal(0.1, 0.05)
        print("Running MCMC to simulate the profit margin...")
        with pm.Model() as mcmc_model:
            margin = pm.Normal("margin", mu=0.1, sigma=0.05)
            # (An error term could be added if you wish to model noise.)
            trace = pm.sample(2000, tune=1000, target_accept=0.95, return_inferencedata=False, progressbar=True)

        # We use the mean of the sampled margin as our estimated profit margin.
        estimated_margin = np.mean(trace["margin"])
        print(f"Estimated profit margin from MCMC: {estimated_margin:.4f}")
        # ---------------------------------------------------

        # Create the Simulated_Profit_Mean column: profit = sale price * estimated_margin
        sim_data['Simulated_Profit_Mean'] = sim_data['SALE PRICE'] * estimated_margin
    else:
        raise KeyError("Neither 'Simulated_Profit_Mean' nor 'SALE PRICE' column found in simulated data.")

# IMPORTANT: Drop any rows that have NaN in the coordinate or profit columns
sim_data_clean = sim_data.dropna(subset=['LON_NBHD', 'LAT_NBHD', 'Simulated_Profit_Mean'])
if sim_data_clean.empty:
    raise ValueError("After dropping rows with NaN coordinates or profit, no data remains for interpolation.")

# Create a grid over the range of the clean simulated data coordinates.
grid_lon = np.linspace(sim_data_clean['LON_NBHD'].min(), sim_data_clean['LON_NBHD'].max(), 100)
grid_lat = np.linspace(sim_data_clean['LAT_NBHD'].min(), sim_data_clean['LAT_NBHD'].max(), 100)
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

# Interpolate the simulated profit over the grid using cubic interpolation.
grid_profit = griddata(
    (sim_data_clean['LON_NBHD'], sim_data_clean['LAT_NBHD']),
    sim_data_clean['Simulated_Profit_Mean'],
    (grid_lon, grid_lat),
    method='cubic'
)

# Plot the fitted profit surface
plt.figure(figsize=(10, 8))
plt.contourf(grid_lon, grid_lat, grid_profit, cmap='coolwarm')
plt.colorbar(label='Simulated Mean Profit')
plt.title('Program 1: MCMC-based Simulated Profit Surface across NYC')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

############################################
# PROGRAM 2: Calculate the Integral of Profit over the Selected Neighborhood
############################################

# Use the geocoded center from Step 1 if available.
if mask_selected.sum() > 0 and 'lat' in locals() and 'lon' in locals():
    center_lat = lat
    center_lon = lon

    # Define a small bounding box around the center (e.g., ±0.01 degrees)
    lat_range = (center_lat - 0.01, center_lat + 0.01)
    lon_range = (center_lon - 0.01, center_lon + 0.01)

    # Create a mask for grid points within this bounding box
    mask_bbox = ((grid_lat >= lat_range[0]) & (grid_lat <= lat_range[1]) &
                 (grid_lon >= lon_range[0]) & (grid_lon <= lon_range[1]))
    profit_neighborhood = grid_profit[mask_bbox]

    # Calculate grid spacings (assuming a uniform grid)
    unique_lats = np.unique(grid_lat[mask_bbox])
    unique_lons = np.unique(grid_lon[mask_bbox])
    if len(unique_lats) > 1 and len(unique_lons) > 1:
        dlat = np.abs(unique_lats[1] - unique_lats[0])
        dlon = np.abs(unique_lons[1] - unique_lons[0])
        area_element = dlat * dlon
    else:
        area_element = 1.0  # Fallback

    # Numerically integrate profit over the bounding box
    total_integrated_profit = np.nansum(profit_neighborhood) * area_element
    print(
        f"Program 2: Total integrated expected profit over '{selected_neighborhood}' is: ${total_integrated_profit:.2f}\n")
else:
    print(f"Program 2: Selected neighborhood '{selected_neighborhood}' is not available for integration.\n")

############################################
# PROGRAM 3: Estimate Price Gradients Geographically from Sales Data
############################################

# Load the updated sales CSV (which now contains geocoded data)
data_sales = pd.read_csv(updated_sales_csv)

# Clean the 'SALE PRICE' column: remove currency symbols and convert to numeric.
data_sales['SALE PRICE'] = pd.to_numeric(
    data_sales['SALE PRICE'].astype(str).str.replace(r'[\$,]', '', regex=True),
    errors='coerce'
)

# Drop rows with missing or invalid data (using geocoded columns)
data_sales = data_sales.dropna(subset=['SALE PRICE', 'LAT_NBHD', 'LON_NBHD'])
data_sales = data_sales[data_sales['SALE PRICE'] > 0]

# Create a grid for geographic interpolation using geocoded columns.
grid_lon2 = np.linspace(data_sales['LON_NBHD'].min(), data_sales['LON_NBHD'].max(), 100)
grid_lat2 = np.linspace(data_sales['LAT_NBHD'].min(), data_sales['LAT_NBHD'].max(), 100)
grid_lon2, grid_lat2 = np.meshgrid(grid_lon2, grid_lat2)

# Interpolate sale prices onto the grid.
grid_price = griddata(
    (data_sales['LON_NBHD'], data_sales['LAT_NBHD']),
    data_sales['SALE PRICE'],
    (grid_lon2, grid_lat2),
    method='linear'
)

# Compute spatial gradients using numpy.gradient.
unique_lat_vals = np.unique(grid_lat2)
unique_lon_vals = np.unique(grid_lon2)
dlat2 = unique_lat_vals[1] - unique_lat_vals[0] if len(unique_lat_vals) > 1 else 1.0
dlon2 = unique_lon_vals[1] - unique_lon_vals[0] if len(unique_lon_vals) > 1 else 1.0

grad_lat, grad_lon = np.gradient(grid_price, dlat2, dlon2)
grad_magnitude = np.sqrt(grad_lat ** 2 + grad_lon ** 2)

# Plot the geographic gradient magnitude of sale prices
plt.figure(figsize=(10, 8))
plt.contourf(grid_lon2, grid_lat2, grad_magnitude, levels=50, cmap='viridis')
plt.colorbar(label='Price Gradient Magnitude')
plt.title('Program 3: Geographic Gradient of Sale Prices in NYC')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
