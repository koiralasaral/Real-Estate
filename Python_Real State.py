import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import pymc as pm
import folium
import branca.colormap as cm
import matplotlib.ticker as mtick
import pytensor
pytensor.config.cxx = ""

def main():
    ############################################
    # STEP 1: Load CSV and Limit to 15 Neighborhoods + Geocoding
    ############################################

    # Path to your original sales CSV; adjust as needed.
    sales_csv = r"C:\Users\LENOVO\Downloads\archive(3)\nyc-rolling-sales.csv"
    data = pd.read_csv(sales_csv)

    # Standardize the neighborhood names.
    data['NEIGHBORHOOD'] = data['NEIGHBORHOOD'].astype(str).str.upper().str.strip()

    # Limit the analysis to the first 15 unique neighborhoods.
    selected_nbhd_list = list(data['NEIGHBORHOOD'].unique()[:15])
    data = data[data['NEIGHBORHOOD'].isin(selected_nbhd_list)]

    # Create empty columns to hold geocoded latitude and longitude.
    data['LAT_NBHD'] = None
    data['LON_NBHD'] = None

    # Initialize the geolocator with increased timeout.
    geolocator = Nominatim(user_agent="nyc_neighborhood_locator", timeout=10)
    # Set a delay of 2 seconds between requests.
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=2)

    neighborhood_coords = {}
    print("Geocoding the following 15 neighborhoods:")
    for nbhd in selected_nbhd_list:
        query = f"{nbhd}, New York, NY"
        try:
            location = geocode(query, timeout=10)
            if location:
                neighborhood_coords[nbhd] = (location.latitude, location.longitude)
                print(f"  {nbhd}: {location.latitude}, {location.longitude}")
            else:
                neighborhood_coords[nbhd] = (np.nan, np.nan)
                print(f"  {nbhd}: Not found")
        except Exception as e:
            print(f"  Error geocoding {nbhd}: {e}")
            neighborhood_coords[nbhd] = (np.nan, np.nan)

    # Map the geocoded coordinates back to each row.
    data['LAT_NBHD'] = data['NEIGHBORHOOD'].map(lambda nb: neighborhood_coords.get(nb, (np.nan, np.nan))[0])
    data['LON_NBHD'] = data['NEIGHBORHOOD'].map(lambda nb: neighborhood_coords.get(nb, (np.nan, np.nan))[1])

    # Save the updated CSV.
    updated_sales_csv = 'nyc-rolling-sales-with-15-neighborhoods-coords.csv'
    data.to_csv(updated_sales_csv, index=False)
    print(f"\nUpdated CSV saved as '{updated_sales_csv}'.")

    ############################################
    # STEP 2: Data Cleaning & Simulate Observed Profit
    ############################################

    # Clean the 'SALE PRICE' field: remove "$" and commas.
    data['SALE PRICE'] = pd.to_numeric(
        data['SALE PRICE'].astype(str).str.replace(r'[\$,]', '', regex=True),
        errors='coerce'
    )
    data = data.dropna(subset=['SALE PRICE', 'LAT_NBHD', 'LON_NBHD'])

    # Simulate an observed profit:
    #   observed_profit = SALE PRICE * 0.10 + noise
    # where noise is drawn from a normal distribution at 2% scale of the sale price.
    np.random.seed(42)  # for reproducibility
    data['observed_profit'] = data['SALE PRICE'] * 0.10 + np.random.normal(0, 1, len(data)) * (
                data['SALE PRICE'] * 0.02)

    # Map neighborhood names to index numbers for hierarchical modeling.
    neighborhoods = data['NEIGHBORHOOD'].unique()
    n_neighborhoods = len(neighborhoods)
    nbhd_to_idx = {nb: i for i, nb in enumerate(neighborhoods)}
    data['nbhd_idx'] = data['NEIGHBORHOOD'].map(nbhd_to_idx)

    ############################################
    # STEP 3: Hierarchical Bayesian Inference (PyMC3)
    ############################################

    print("\nRunning hierarchical Bayesian inference...")
    with pm.Model() as hierarchical_model:
        # Global hyper-priors for the profit margin.
        mu_margin = pm.Normal("mu_margin", mu=0.10, sigma=0.05)  # overall mean margin
        sigma_margin = pm.HalfNormal("sigma_margin", sigma=0.05)

        # Neighborhood-specific profit margins.
        margin = pm.Normal("margin", mu=mu_margin, sigma=sigma_margin, shape=n_neighborhoods)

        # Standard deviation for the observational error.
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=1e5)

        # Expected profit for each sale:
        mu_profit = data['SALE PRICE'].values * margin[data['nbhd_idx'].values]

        # Likelihood: the observed profit.
        observed = pm.Normal("observed", mu=mu_profit, sigma=sigma_obs,
                             observed=data['observed_profit'].values)

        # Sample from the posterior.
        trace = pm.sample(2000, tune=1000, target_accept=0.95, return_inferencedata=False)

    # Compute posterior summaries for each neighborhood's profit margin.
    posterior_margin_mean = trace["margin"].mean(axis=0)
    posterior_margin_hpd = pm.stats.hpd(trace["margin"], hdi_prob=0.90)

    ############################################
    # STEP 4: Matplotlib Visualization (Bar Plot)
    ############################################

    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(n_neighborhoods), posterior_margin_mean,
                   yerr=[posterior_margin_mean - posterior_margin_hpd[:, 0],
                         posterior_margin_hpd[:, 1] - posterior_margin_mean],
                   capsize=5, color='skyblue', alpha=0.8)
    plt.xticks(range(n_neighborhoods), neighborhoods, rotation=90)
    plt.ylabel("Estimated Profit Margin")
    plt.title("Posterior Estimated Profit Margin by Neighborhood")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.tight_layout()
    plt.show()

    ############################################
    # STEP 5: Folium Map Visualization
    ############################################

    # Create a Folium map centered on New York City.
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=11)

    # Create a linear colormap for the estimated margin.
    min_margin = posterior_margin_mean.min()
    max_margin = posterior_margin_mean.max()
    colormap = cm.LinearColormap(colors=['blue', 'green', 'yellow', 'red'],
                                 vmin=min_margin, vmax=max_margin,
                                 caption='Estimated Profit Margin')

    # Add markers for each neighborhood.
    for nbhd in neighborhoods:
        # Get a representative row for this neighborhood.
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

    # Add the colormap legend to the map.
    colormap.add_to(m)
    folium_map_file = "nyc_profit_margin_map_15nbhd.html"
    m.save(folium_map_file)
    print(f"\nFolium map saved as '{folium_map_file}'.")


if __name__ == '__main__':
    main()
