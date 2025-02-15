# Health Statistics Analysis Application

## Overview
This application analyzes health data to generate comprehensive reports and visualizations of patient vital signs. It processes medical data to identify trends, abnormal readings, and statistical patterns in blood pressure, heart rate, and age demographics.

## Features
- Statistical analysis of vital signs
- Visualization of health trends over time
- Abnormal reading detection and reporting
- Distribution analysis of vital measurements
- Correlation studies between different health metrics

## Getting Started

### Prerequisites
- Python 3.x
- Required packages: pandas, numpy, matplotlib, seaborn, tabulate

### Installation
1. Clone the repository
2. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn tabulate


# Sentiment Analysis with Logistic Regression

This repository provides a sentiment analysis implementation using Logistic Regression, designed to answer the following **Business Analytics** question:

### Question:
**Create a program that performs sentiment analysis on customer reviews for a product or service. The program should read a set of customer reviews, analyze the sentiment (positive, negative, neutral), and provide a summary of customer feedback along with any notable recurring themes.**

### Requirements:
- The program outputs a **sentiment analysis report** that includes the percentage of positive, negative, and neutral sentiments, as well as **key themes** extracted from the reviews.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Setup Instructions](#setup-instructions)
- [How It Works](#how-it-works)
- [Model Accuracy](#model-accuracy)
- [Sentiment Analysis Report](#sentiment-analysis-report)
- [License](#license)

## Overview

This project leverages logistic regression to classify sentiment in Amazon product reviews. The pipeline consists of the following steps:

1. **Text Cleaning**: Removes special characters, stopwords, and converts the text to lowercase.
2. **Feature Extraction**: Converts text into numeric vectors using TF-IDF (Term Frequency-Inverse Document Frequency).
3. **Model Training**: Uses Logistic Regression to train on the training data.
4. **Sentiment Classification**: Classifies reviews into positive or negative sentiments.
5. **Sentiment Analysis Report**: Outputs sentiment percentages and key themes based on TF-IDF scores.

The code was built specifically to address the question about performing sentiment analysis and summarizing customer feedback for business analytics purposes.

## Requirements

To run this project, you need the following libraries:

- `pandas`
- `nltk`
- `scikit-learn`
- `numpy`

You can install these dependencies using pip:

```bash
pip install pandas nltk scikit-learn numpy
```

Make sure you also download the `stopwords` corpus from NLTK by running the following in your Python script:

```python
import nltk
nltk.download('stopwords')
```

## Dataset

The dataset used in this project comes from the Kaggle dataset: [Amazon Product Reviews](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews/data). The dataset consists of two `.txt` files:

- `train.ft.txt`: The training data (contains product reviews and their corresponding sentiment labels).
- `test.ft.txt`: The testing data (used to evaluate the model's performance).

You will need to download the dataset from the Kaggle link and place the `.txt` files in your project directory.

## Setup Instructions

1. **Download the Dataset**:
   - Go to the [Amazon Product Reviews dataset](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews/data).
   - Download the `train.ft.txt` and `test.ft.txt` files.
   - Place them in your project directory (or update the paths in the code accordingly).

2. **Run the Script**:
   After setting up your environment and dataset, you can run the script by executing:

   ```bash
   python sentiment_analysis.py
   ```

3. **Interpret the Results**:
   - The script will output the model accuracy.
   - It will also show the sentiment percentages (positive, negative, and neutral) for the reviews in the test data.
   - Finally, it will print the top 10 key themes/words extracted from the training data based on TF-IDF scores.

## How It Works

1. **Data Preprocessing**:
   - **Text Cleaning**: The reviews are cleaned by:
     - Removing special characters and numbers.
     - Converting text to lowercase.
     - Removing stopwords (common words like "the", "and", "is", etc.).
   
   ```python
   def clean_text(text):
       text = re.sub(r"[^a-zA-Z\s]", "", text.lower())  # Remove special characters & convert to lowercase
       words = text.split()
       words = [word for word in words if word not in stop_words]
       return " ".join(words)
   ```

2. **Feature Extraction**:
   - We use the **TF-IDF Vectorizer** to convert the cleaned text into numeric vectors that the machine learning model can understand. 
   - The `TfidfVectorizer` converts each word into a unique number based on its frequency and importance in the text corpus.

   ```python
   vectorizer = TfidfVectorizer(max_features=5000)
   X_train = vectorizer.fit_transform(df_train["clean_reviews"])
   X_test = vectorizer.transform(df_test["clean_reviews"])
   ```

3. **Logistic Regression Model**:
   - We train a **Logistic Regression** model on the training data. Logistic regression is suitable for binary classification tasks (positive/negative sentiment in this case).
   
   ```python
   model = LogisticRegression()
   model.fit(X_train, y_train)
   ```

4. **Sentiment Analysis**:
   - The model predicts whether a review has a positive or negative sentiment. 
   - The accuracy of the model is evaluated on the test data.
   - We also calculate the percentage of positive, negative, and neutral sentiments in the test dataset.

   ```python
   accuracy = accuracy_score(y_test, y_pred)
   ```

5. **Key Themes Extraction**:
   - Using TF-IDF, the script identifies the most important words (key themes) in the reviews.

   ```python
   vectorizer = TfidfVectorizer(max_features=10) # Top 10 words
   words = np.array(vectorizer.get_feature_names_out())
   tfidf_scores = np.array(X.sum(axis=0)).flatten()
   top_words = [words[i] for i in tfidf_scores.argsort()[-10:][::-1]]
   ```

## Model Accuracy

Once the model has been trained, the accuracy is printed as follows:

```python
accuracy = accuracy_score(y_test, y_pred)  
print(f"Model Accuracy: {accuracy:.4f}")
```

## Sentiment Analysis Report

After the model is trained, the script will output a report with the following information:

- **Sentiment Percentages**: Percentage of positive, negative, and neutral sentiments in the test data.
- **Top 10 Key Themes**: The 10 most important words (key themes) based on TF-IDF scores extracted from the training data.

```python
print(f"Positive Sentiment: {positive_percentage:.2f}%")
print(f"Negative Sentiment: {negative_percentage:.2f}%")
print(f"Neutral Sentiment: {neutral_percentage:.2f}%")
print("\nTop 10 Key Themes/Words in Reviews:")
for word, score in zip(top_words, top_scores):
    print(f"{word}: {score:.4f}")
```

## License

This project is licensed under the MIT License.

---


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
