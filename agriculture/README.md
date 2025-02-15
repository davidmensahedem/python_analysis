# Irrigation Recommendation System

## Overview
This application assists farmers in making irrigation decisions based on soil moisture data. It analyzes current soil moisture levels and historical weather data to provide recommendations on whether irrigation is necessary. The system considers factors like crop type and local weather patterns to offer insights on soil moisture trends and historical weather patterns.

## Features
- **Soil Moisture Analysis**: Analyzes soil moisture data from multiple sensors.
- **Irrigation Recommendations**: Provides recommendations on whether to irrigate based on predefined thresholds.
- **Historical Data Comparison**: Compares current moisture levels with historical data to offer additional insights.
- **Interactive Input**: Allows users to input current soil moisture levels for real-time recommendations.

## Dataset Structure
To use this application, your dataset should be structured as follows:

### Required Columns
- **year**: The year of the data recording (e.g., 2020).
- **month**: The month of the data recording (e.g., 3 for March).
- **day**: The day of the data recording (e.g., 6).
- **hour**: The hour of the data recording (e.g., 22).
- **minute**: The minute of the data recording (e.g., 16).
- **second**: The second of the data recording (e.g., 11).
- **moisture0**: Soil moisture reading from sensor 0 (e.g., 0.70).
- **moisture1**: Soil moisture reading from sensor 1 (e.g., 0.64).
- **moisture2**: Soil moisture reading from sensor 2 (e.g., 0.73).
- **moisture3**: Soil moisture reading from sensor 3 (e.g., 0.40).
- **moisture4**: Soil moisture reading from sensor 4 (e.g., 0.02).
- **irrgation**: Boolean indicating whether irrigation was applied (e.g., False).

### Example Dataset
```csv
year,month,day,hour,minute,second,moisture0,moisture1,moisture2,moisture3,moisture4,irrgation
2020,3,6,22,16,11,0.70,0.64,0.73,0.40,0.02,False
2020,3,6,22,17,11,0.70,0.64,0.71,0.39,0.02,False
2020,3,6,22,18,11,0.69,0.63,0.70,0.39,0.02,False
```

## Usage

### Prerequisites
- Python 3.x
- Pandas
- Matplotlib

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/irrigation-recommendation-system.git
   cd irrigation-recommendation-system
   ```
2. Install the required packages:
   ```bash
   pip install pandas matplotlib
   ```

### Running the Application
1. Place your dataset CSV files in the project directory.
2. Update the file paths in the script if necessary.
3. Run the script:
   ```bash
   python irrigation_recommendation.py
   ```

### Interactive Input
When prompted, enter the current soil moisture levels for each sensor:
```
Enter moisture0 level (0.0 to 1.0): 0.4
Enter moisture1 level (0.0 to 1.0): 0.5
Enter moisture2 level (0.0 to 1.0): 0.6
Enter moisture3 level (0.0 to 1.0): 0.3
Enter moisture4 level (0.0 to 1.0): 0.2
```

### Output
The application will provide:
- **Recommendation**: Whether irrigation is recommended.
- **Insights**: Average soil moisture, threshold, and moisture readings.
- **Historical Records**: Similar historical records and whether irrigation was applied.

## Example Output
```
=== Recommendation ===
Irrigation recommended

=== Insights ===
Average soil moisture: 0.40
Threshold: 0.5
Moisture readings: [0.4, 0.5, 0.6, 0.3, 0.2]

=== Similar Historical Records ===
Found 12 similar records.
   year  month  day  hour  minute  second  irrgation
0  2020      3    6    22      16      11      False
1  2020      3    6    22      17      11      False
2  2020      3    6    22      18      11      False
3  2020      3    6    22      19      11      False
4  2020      3    6    22      20      12      False
```

## Customization
- **Moisture Threshold**: Adjust the threshold in the `recommend_irrigation` function based on crop requirements.
- **Historical Data Comparison**: Modify the `find_similar_records` function to change the similarity threshold.

## Support
For any issues or questions, please open an issue on the GitHub repository or contact the maintainers.

## License
This project is licensed under the MIT License. See the LICENSE file for details.