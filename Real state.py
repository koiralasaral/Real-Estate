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
df.fillna(0, inplace=True)

# Ensure data is sorted by date for trend analysis
df.sort_values(by='SALE DATE', inplace=True)

# --- 1. Trend Analysis (Discrete Rate of Change) ---

# Example 1.1: Monthly Change in Average Sale Price
monthly_avg_price = df.groupby(df['SALE DATE'].dt.to_period('M'))['SALE PRICE'].mean()
monthly_price_change = monthly_avg_price.pct_change() * 100

print("\nMonthly Percentage Change in Average Sale Price:")
print(monthly_price_change.head())

plt.figure(figsize=(12, 6))
monthly_price_change.plot(kind='line', marker='o')
plt.title('Monthly Percentage Change in Average Sale Price')
plt.xlabel('Month-Year')
plt.ylabel('% Change')
plt.grid(True)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
# Analysis: Shows the rate at which the average sale price is changing month to month. Volatility indicates rapid changes.

# Example 1.2: Yearly Change in the Number of Sales
yearly_sales_count = df.groupby(df['SALE DATE'].dt.year).size()
yearly_sales_change = yearly_sales_count.pct_change() * 100

print("\nYearly Percentage Change in Number of Sales:")
print(yearly_sales_change)

plt.figure(figsize=(10, 6))
yearly_sales_change.plot(kind='bar')
plt.title('Yearly Percentage Change in Number of Sales')
plt.xlabel('Year')
plt.ylabel('% Change')
plt.grid(True)
plt.show()
# Analysis: Illustrates the growth or decline rate of the number of transactions annually.

# Example 1.3: Rolling Average of Sale Price (Smoothing Trends)
# A rolling average helps to smooth out short-term fluctuations and reveal longer-term trends.
rolling_avg_price = df['SALE PRICE'].rolling(window=30).mean() # 30-day rolling average

plt.figure(figsize=(14, 7))
plt.plot(df['SALE DATE'], df['SALE PRICE'], alpha=0.4, label='Daily Sale Price')
plt.plot(df['SALE DATE'], rolling_avg_price, color='red', label='30-Day Rolling Average')
plt.title('Daily Sale Price vs. 30-Day Rolling Average')
plt.xlabel('Date')
plt.ylabel('Sale Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# Analysis: The rolling average line smooths the daily price fluctuations, making it easier to see the overall trend (increasing, decreasing, or stable).

# --- 2. Accumulation Analysis (Discrete Cumulative Sums) ---

# Example 2.1: Cumulative Number of Sales Over Time (Revisited)
cumulative_sales = df.groupby(df['SALE DATE'].dt.to_period('M')).size().cumsum()

print("\nCumulative Number of Sales Over Time:")
print(cumulative_sales.head())

plt.figure(figsize=(12, 6))
cumulative_sales.plot(kind='line', marker='o')
plt.title('Cumulative Number of Sales Over Time')
plt.xlabel('Month-Year')
plt.ylabel('Cumulative Sales')
plt.grid(True)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
# Analysis: Shows the total number of sales that have occurred up to each point in time. The slope indicates the rate of sales activity.

# Example 2.2: Cumulative Sum of Sale Prices Over Time
# This shows the total value of real estate sold up to a certain point.
cumulative_value = df.groupby(df['SALE DATE'].dt.to_period('M'))['SALE PRICE'].sum().cumsum()

print("\nCumulative Sum of Sale Prices Over Time:")
print(cumulative_value.head())

plt.figure(figsize=(12, 6))
cumulative_value.plot(kind='line', marker='o', color='green')
plt.title('Cumulative Sum of Sale Prices Over Time')
plt.xlabel('Month-Year')
plt.ylabel('Cumulative Sale Value')
plt.grid(True)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
# Analysis: Indicates the total monetary value of transactions over time. The slope reflects the rate at which value is being transacted.

# Example 2.3: Cumulative Units Sold Over Time (Residential and Commercial)
cumulative_residential_units = df.groupby(df['SALE DATE'].dt.to_period('M'))['RESIDENTIAL UNITS'].sum().cumsum()
cumulative_commercial_units = df.groupby(df['SALE DATE'].dt.to_period('M'))['COMMERCIAL UNITS'].sum().cumsum()

plt.figure(figsize=(12, 6))
plt.plot(cumulative_residential_units.index.to_timestamp(), cumulative_residential_units, label='Cumulative Residential Units', marker='o')
plt.plot(cumulative_commercial_units.index.to_timestamp(), cumulative_commercial_units, label='Cumulative Commercial Units', marker='o')
plt.title('Cumulative Units Sold Over Time')
plt.xlabel('Month-Year')
plt.ylabel('Cumulative Units')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
# Analysis: Tracks the total number of residential and commercial units sold over time, showing their respective accumulation trends.

# --- 3. Combining Trend and Accumulation ---

# Example 3.1: Rate of Change of Cumulative Sales
# We can look at the month-over-month change in the cumulative sales to see how the rate of accumulation is changing.
rate_of_cumulative_sales = cumulative_sales.diff()

print("\nMonth-over-Month Change in Cumulative Sales:")
print(rate_of_cumulative_sales.head())

plt.figure(figsize=(12, 6))
rate_of_cumulative_sales.plot(kind='line', marker='o', color='purple')
plt.title('Month-over-Month Change in Cumulative Sales (Rate of Accumulation)')
plt.xlabel('Month-Year')
plt.ylabel('Change in Cumulative Sales')
plt.grid(True)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
# Analysis: This shows the number of sales in each month. An increasing trend here means the rate of accumulation of total sales is increasing.

# Example 3.2: Comparing Rate of Price Change with Cumulative Value
# We can plot the monthly price change alongside the cumulative sale value to see if there's a relationship.

plt.figure(figsize=(14, 7))
plt.plot(monthly_price_change.index.to_timestamp(), monthly_price_change, label='Monthly % Change in Avg Price', color='blue', linestyle='--')
plt.twinx() # Create a second y-axis
plt.plot(cumulative_value.index.to_timestamp(), cumulative_value, label='Cumulative Sale Value', color='green')
plt.ylabel('Cumulative Sale Value', color='green')
plt.xlabel('Month-Year')
plt.title('Monthly Price Change vs. Cumulative Sale Value')
plt.legend(loc='upper left')
plt.gca().yaxis.label.set_color('green')
plt.grid(True)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
# Analysis: This combined plot can help identify if changes in the rate of price increase/decrease correlate with the total value of transactions.

# ... you can continue with more examples focusing on different variables ...