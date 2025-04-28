import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
try:
    df = pd.read_csv("C:\\Users\\LENOVO\\Downloads\\archive(3)\\nyc-rolling-sales.csv")  # Replace "your_file.csv" with the actual file name
except FileNotFoundError:
    print("Error: CSV file not found.")
    exit()

# Data Cleaning and Preprocessing
df['SALE PRICE'] = pd.to_numeric(df['SALE PRICE'], errors='coerce')
df['SALE DATE'] = pd.to_datetime(df['SALE DATE'])
df['LAND SQUARE FEET'] = pd.to_numeric(df['LAND SQUARE FEET'], errors='coerce').fillna(0)
df['GROSS SQUARE FEET'] = pd.to_numeric(df['GROSS SQUARE FEET'], errors='coerce').fillna(0)
df.fillna(0, inplace=True)

# Ensure data is sorted by date
df.sort_values(by='SALE DATE', inplace=True)

# --- Program 1: Trend in Building Class Categories ---
# Analyze the yearly percentage change in the number of sales for the top N building classes.
top_n = 5
yearly_building_class_counts = df.groupby([df['SALE DATE'].dt.year, 'BUILDING CLASS CATEGORY']).size().unstack(fill_value=0)
top_building_classes = yearly_building_class_counts.sum().nlargest(top_n).index
yearly_building_class_counts_top = yearly_building_class_counts[top_building_classes]
yearly_building_class_change = yearly_building_class_counts_top.pct_change() * 100

print("\n--- Trend in Top {} Building Class Categories (Yearly % Change) ---".format(top_n))
print(yearly_building_class_change)

plt.figure(figsize=(14, 7))
yearly_building_class_change.plot(kind='line', marker='o')
plt.title('Yearly Percentage Change in Number of Sales by Top {} Building Class'.format(top_n))
plt.xlabel('Year')
plt.ylabel('% Change')
plt.legend(title='Building Class Category')
plt.grid(True)
plt.show()

# --- Program 2: Accumulation of Square Footage Sold ---
# Analyze the cumulative sum of land and gross square footage sold over time.
monthly_sqft_sold = df.groupby(df['SALE DATE'].dt.to_period('M'))[['LAND SQUARE FEET', 'GROSS SQUARE FEET']].sum().cumsum()

print("\n--- Accumulation of Square Footage Sold (Monthly Cumulative Sum) ---")
print(monthly_sqft_sold.head())

plt.figure(figsize=(12, 6))
plt.plot(monthly_sqft_sold.index.to_timestamp(), monthly_sqft_sold['LAND SQUARE FEET'], label='Cumulative Land Square Feet', marker='o')
plt.plot(monthly_sqft_sold.index.to_timestamp(), monthly_sqft_sold['GROSS SQUARE FEET'], label='Cumulative Gross Square Feet', marker='o')
plt.title('Cumulative Square Footage Sold Over Time')
plt.xlabel('Month-Year')
plt.ylabel('Cumulative Square Feet')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# --- Program 3: Trend in Average Price Per Square Foot ---
# Analyze the monthly trend in the average price per land square foot and gross square foot.
df['PRICE PER LAND SQFT'] = df['SALE PRICE'] / (df['LAND SQUARE FEET'] + 1e-6) # Avoid division by zero
df['PRICE PER GROSS SQFT'] = df['SALE PRICE'] / (df['GROSS SQUARE FEET'] + 1e-6) # Avoid division by zero

monthly_avg_price_per_sqft = df.groupby(df['SALE DATE'].dt.to_period('M'))[['PRICE PER LAND SQFT', 'PRICE PER GROSS SQFT']].mean()

print("\n--- Trend in Average Price Per Square Foot (Monthly) ---")
print(monthly_avg_price_per_sqft.head())

plt.figure(figsize=(12, 6))
monthly_avg_price_per_sqft.plot(kind='line', marker='o')
plt.title('Monthly Trend in Average Price Per Square Foot')
plt.xlabel('Month-Year')
plt.ylabel('Average Price per Sqft')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# --- Program 4: Rate of Change by Neighborhood ---
# Analyze the yearly percentage change in the average sale price for the top N neighborhoods.
top_n_neighborhoods = 5
yearly_avg_price_by_neighborhood = df.groupby([df['SALE DATE'].dt.year, 'NEIGHBORHOOD'])['SALE PRICE'].mean().unstack(fill_value=0)
top_neighborhoods = yearly_avg_price_by_neighborhood.sum().nlargest(top_n_neighborhoods).index
yearly_avg_price_by_neighborhood_top = yearly_avg_price_by_neighborhood[top_neighborhoods]
yearly_avg_price_change_by_neighborhood = yearly_avg_price_by_neighborhood_top.pct_change() * 100

print("\n--- Rate of Change in Average Sale Price by Top {} Neighborhoods (Yearly % Change) ---".format(top_n_neighborhoods))
print(yearly_avg_price_change_by_neighborhood)

plt.figure(figsize=(14, 7))
yearly_avg_price_change_by_neighborhood.plot(kind='line', marker='o')
plt.title('Yearly Percentage Change in Average Sale Price by Top {} Neighborhoods'.format(top_n_neighborhoods))
plt.xlabel('Year')
plt.ylabel('% Change')
plt.legend(title='Neighborhood')
plt.grid(True)
plt.show()