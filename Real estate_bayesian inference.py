import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import pymc as pm  # Make sure PyMC3 is installed
import pytensor  # Suppressing g++ warnings

# Suppress g++ warnings to avoid performance degradation messages
pytensor.config.cxx = ""

############################################
# SETTINGS & INPUTS
############################################

# Specify the selected neighborhood (change as needed)
selected_neighborhood = "ALPHABET CITY"

# CSV filenames (adjust the paths if needed)
sales_csv = r"C:\Users\LENOVO\Downloads\archive(3)\nyc-rolling-sales.csv"  # Original sales data file
simulated_csv = "nyc-rolling-sales-with-selected-neighborhood-coords.csv"  # File to use for simulation

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

# Use the safe multiprocessing idiom
if __name__ == "__main__":
    # Load simulated profit data from the updated CSV file.
    sim_data = pd.read_csv(simulated_csv)

    # Generate simulated profit using an MCMC method if not already present.
    if 'Simulated_Profit_Mean' not in sim_data.columns:
        if 'SALE PRICE' in sim_data.columns:
            # Clean the SALE PRICE column: remove currency symbols and commas
            sim_data['SALE PRICE'] = pd.to_numeric(
                sim_data['SALE PRICE'].astype(str).str.replace(r'[\$,]', '', regex=True),
                errors='coerce'
            )

            # --- MCMC Simulation for the profit margin ---
            print("Running MCMC to simulate the profit margin...")
            with pm.Model() as mcmc_model:
                margin = pm.Normal("margin", mu=0.1, sigma=0.05)
                trace = pm.sample(2000, tune=1000, target_accept=0.95, cores=1, return_inferencedata=False, progressbar=True)

            # We use the mean of the sampled margin as our estimated profit margin.
            estimated_margin = np.mean(trace["margin"])
            print(f"Estimated profit margin from MCMC: {estimated_margin:.4f}")

            # Create the Simulated_Profit_Mean column: profit = sale price * estimated_margin
            sim_data['Simulated_Profit_Mean'] = sim_data['SALE PRICE'] * estimated_margin
        else:
            raise KeyError("Neither 'Simulated_Profit_Mean' nor 'SALE PRICE' column found in simulated data.")

    # IMPORTANT: Drop any rows that have NaN in the coordinate or profit columns
    sim_data_clean = sim_data.dropna(subset=['LON_NBHD', 'LAT_NBHD', 'Simulated_Profit_Mean'])
    if sim_data_clean.empty:
        raise ValueError("After dropping rows with NaN coordinates or profit, no data remains for interpolation.")

    # Ensure there is sufficient unique data for interpolation
    print("Validating input data...")
    if len(sim_data_clean) < 3:
        raise ValueError("Insufficient unique points for interpolation. Please provide more diverse data.")

    # Add jitter if data points are nearly identical
    sim_data_clean['LON_NBHD'] += np.random.normal(0, 1e-5, len(sim_data_clean))
    sim_data_clean['LAT_NBHD'] += np.random.normal(0, 1e-5, len(sim_data_clean))

    # Create a grid over the range of the clean simulated data coordinates
    grid_lon = np.linspace(sim_data_clean['LON_NBHD'].min(), sim_data_clean['LON_NBHD'].max(), 100)
    grid_lat = np.linspace(sim_data_clean['LAT_NBHD'].min(), sim_data_clean['LAT_NBHD'].max(), 100)
    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

    # Use griddata for interpolation
    print("Interpolating profit surface...")
    grid_profit = griddata(
        (sim_data_clean['LON_NBHD'], sim_data_clean['LAT_NBHD']),
        sim_data_clean['Simulated_Profit_Mean'],
        (grid_lon, grid_lat),
        method='linear'  # Or 'nearest'
    )

    # Plot the fitted profit surface
    plt.figure(figsize=(10, 8))
    plt.contourf(grid_lon, grid_lat, grid_profit, cmap='coolwarm')
    plt.colorbar(label='Simulated Mean Profit')
    plt.title('Program 1: MCMC-based Simulated Profit Surface across NYC')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()