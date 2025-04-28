# 📓 Berlin Real Estate Statistical Study

# 1. Setup and Imports
import numpy as np
import matplotlib.pyplot as plt
import folium
import scipy.stats as stats

# 2. Sample Data
districts = ['Mitte', 'Kreuzberg', 'Charlottenburg', 'Neukölln', 'Marzahn-Hellersdorf']
area = np.array([75, 60, 80, 65, 90])  # Area Size (m²)
price = np.array([543750, 375000, 540000, 341250, 360000])  # Total Price (EUR)
price_per_m2 = price / area

# Mean (mu) and Standard Deviation (sigma)
mu = np.mean(price_per_m2)
sigma = np.std(price_per_m2)

print("Price per m²:", price_per_m2)
print(f"Estimated Mean (\u03bc): {mu:.2f} €/m²")
print(f"Standard Deviation (\u03c3): {sigma:.2f} €/m²")

# 3. Scatter Plot (Area vs Price per m²)
plt.figure(figsize=(8,6))
plt.scatter(area, price_per_m2, color='red')
plt.title('Area vs Price per m² in Berlin Districts')
plt.xlabel('Area Size (m²)')
plt.ylabel('Price per m² (€)')
plt.grid(True)
plt.show()

# 4. Folium Interactive Map
berlin_coords = [52.52, 13.405]
m = folium.Map(location=berlin_coords, zoom_start=11)

district_coords = {
    'Mitte': [52.5200, 13.4050],
    'Kreuzberg': [52.4996, 13.4030],
    'Charlottenburg': [52.5167, 13.3041],
    'Neukölln': [52.4800, 13.4500],
    'Marzahn-Hellersdorf': [52.5450, 13.6000]
}

for dist, coord in district_coords.items():
    folium.CircleMarker(
        location=coord,
        radius=8,
        popup=f"{dist}: {price_per_m2[districts.index(dist)]:.0f} €/m²",
        color='blue',
        fill=True,
        fill_color='blue'
    ).add_to(m)

m.save('berlin_real_estate_map.html')
print("Folium map saved as 'berlin_real_estate_map.html'. Open it in a browser to view.")

# 5. Distribution Check (Histogram)
plt.figure(figsize=(8,6))
plt.hist(price_per_m2, bins=5, color='skyblue', edgecolor='black', density=True)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, sigma)
plt.plot(x, p, 'k', linewidth=2)
plt.title('Distribution of Price per m²')
plt.xlabel('Price per m² (€)')
plt.ylabel('Density')
plt.grid(True)
plt.show()

# 6. Moment Generating Function (MGF)
t_mgf = np.linspace(-0.001, 0.001, 100)
mgf = np.exp(mu*t_mgf + 0.5*sigma**2*t_mgf**2)

plt.figure(figsize=(8,6))
plt.plot(t_mgf, mgf)
plt.title('Moment Generating Function (MGF)')
plt.xlabel('t')
plt.ylabel('M_X(t)')
plt.grid(True)
plt.show()

# 7. Characteristic Function (CF)
t_cf = np.linspace(-0.005, 0.005, 100)
cf_real = np.cos(mu*t_cf) * np.exp(-0.5 * sigma**2 * t_cf**2)
cf_imag = np.sin(mu*t_cf) * np.exp(-0.5 * sigma**2 * t_cf**2)

plt.figure(figsize=(10,6))
plt.plot(t_cf, cf_real, label='Re(\u03c6(t))')
plt.plot(t_cf, cf_imag, label='Im(\u03c6(t))', linestyle='--')
plt.title('Characteristic Function (CF)')
plt.xlabel('t')
plt.ylabel('φ(t)')
plt.grid(True)
plt.legend()
plt.show()

# 8. Summary
print("\nSUMMARY:")
print(f"Mean (\u03bc): {mu:.2f} €/m²")
print(f"Standard Deviation (\u03c3): {sigma:.2f} €/m²")
print("Distribution: Normal Distribution Approximation")
print("MGF and CF plotted successfully.")
print("Interactive Map saved. Open 'berlin_real_estate_map.html' to view.")
