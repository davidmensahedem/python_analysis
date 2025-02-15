# %%
import pandas as pd
import matplotlib.pyplot as plt


# %%
# List of file paths
file_paths = ['/Users/hughes/Downloads/soil_moisture/plant_vase1.CSV',  '/Users/hughes/Downloads/soil_moisture/plant_vase1(2).CSV', '/Users/hughes/Downloads/soil_moisture/plant_vase2.CSV']
dfs = [pd.read_csv(file) for file in file_paths]

# Combine datasets into one
combined_df = pd.concat(dfs, ignore_index=True)

# Display combined dataset info
print("Combined Dataset Info:")
print(combined_df.info())
print("\nFirst 5 rows:")
print(combined_df.head())


# %%
def recommend_irrigation(moisture_values, threshold=0.5):
    """
    Recommend irrigation based on soil moisture values.
    :param moisture_values: List of moisture readings (e.g., [moisture0, moisture1, moisture2, moisture3, moisture4])
    :param threshold: Moisture threshold below which irrigation is recommended (default: 0.5)
    :return: Recommendation and insights
    """
    avg_moisture = sum(moisture_values) / len(moisture_values)
    recommendation = "Irrigation recommended" if avg_moisture < threshold else "No irrigation needed"
    insights = [
        f"Average soil moisture: {avg_moisture:.2f}",
        f"Threshold: {threshold}",
        f"Moisture readings: {moisture_values}"
    ]
    return recommendation, insights

# %%
# Define moisture threshold (adjust based on crop requirements)
moisture_threshold = 0.5

# Add recommendation column to the dataset
combined_df['recommendation'] = combined_df.apply(
    lambda row: recommend_irrigation([row['moisture0'], row['moisture1'], row['moisture2'], row['moisture3'], row['moisture4']], threshold=moisture_threshold)[0],
    axis=1
)

# Display sample recommendations
print("\nSample Recommendations:")
print(combined_df[['year', 'month', 'day', 'hour', 'minute', 'second', 'recommendation']].head(10))

# %%
# Analyze irrigation recommendations
irrigation_counts = combined_df['recommendation'].value_counts()
print("\nIrrigation Recommendation Counts:")
print(irrigation_counts)

# Calculate average moisture levels over time
combined_df['timestamp'] = pd.to_datetime(combined_df[['year', 'month', 'day', 'hour', 'minute', 'second']])
combined_df['avg_moisture'] = combined_df[['moisture0', 'moisture1', 'moisture2', 'moisture3', 'moisture4']].mean(axis=1)

# Plot moisture trends

plt.figure(figsize=(10, 6))
plt.plot(combined_df['timestamp'], combined_df['avg_moisture'], label='Average Soil Moisture')
plt.axhline(y=moisture_threshold, color='r', linestyle='--', label='Irrigation Threshold')
plt.title('Soil Moisture Trends Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Average Soil Moisture')
plt.legend()
plt.show()

# %%
def interactive_irrigation_recommendation():
    """
    Interactive function to take user inputs and provide irrigation recommendations.
    """
    print("=== Irrigation Recommendation System ===")
    
    # Take user inputs
    try:
        moisture0 = float(input("Enter moisture0 level (0.0 to 1.0): "))
        moisture1 = float(input("Enter moisture1 level (0.0 to 1.0): "))
        moisture2 = float(input("Enter moisture2 level (0.0 to 1.0): "))
        moisture3 = float(input("Enter moisture3 level (0.0 to 1.0): "))
        moisture4 = float(input("Enter moisture4 level (0.0 to 1.0): "))
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return

    # Define moisture threshold (adjust based on crop requirements)
    moisture_threshold = 0.5

    # Get recommendation
    recommendation, insights = recommend_irrigation([moisture0, moisture1, moisture2, moisture3, moisture4], threshold=moisture_threshold)

    # Display results
    print("\n=== Recommendation ===")
    print(recommendation)
    print("\n=== Insights ===")
    for insight in insights:
        print(insight)

    # Find similar historical records
    similar_records = find_similar_records([moisture0, moisture1, moisture2, moisture3, moisture4])
    if not similar_records.empty:
        print("\n=== Similar Historical Records ===")
        print(f"Found {len(similar_records)} similar records.")
        print(similar_records[['year', 'month', 'day', 'hour', 'minute', 'second', 'irrgation']].head())
    else:
        print("\nNo similar historical records found.")

# %%
def find_similar_records(moisture_values, threshold=0.1):
    """
    Find historical records with similar moisture levels.
    :param moisture_values: List of moisture readings (e.g., [moisture0, moisture1, moisture2, moisture3, moisture4])
    :param threshold: Maximum difference to consider records similar (default: 0.1)
    :return: DataFrame of similar records
    """
    # Calculate absolute difference for each moisture column
    for i, col in enumerate(['moisture0', 'moisture1', 'moisture2', 'moisture3', 'moisture4']):
        combined_df[f'diff_{col}'] = abs(combined_df[col] - moisture_values[i])
    
    # Filter records where all differences are within the threshold
    similar_records = combined_df[
        (combined_df['diff_moisture0'] <= threshold) &
        (combined_df['diff_moisture1'] <= threshold) &
        (combined_df['diff_moisture2'] <= threshold) &
        (combined_df['diff_moisture3'] <= threshold) &
        (combined_df['diff_moisture4'] <= threshold)
    ]
    
    # Drop difference columns
    similar_records = similar_records.drop(columns=[f'diff_{col}' for col in ['moisture0', 'moisture1', 'moisture2', 'moisture3', 'moisture4']])
    
    return similar_records

# %%
# Run the interactive function
# interactive_irrigation_recommendation()

# %%



