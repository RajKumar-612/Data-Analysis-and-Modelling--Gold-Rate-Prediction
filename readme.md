# Gold Price Prediction

This project analyzes and predicts the price of Gold ETF using various financial indicators and machine learning models.

## Project Structure

- **Importing Libraries**: Import necessary libraries for data manipulation, visualization, and modeling.
- **Loading Data**: Load the dataset containing historical gold prices and other financial indicators.
- **Data Exploration**: Explore the dataset to understand its structure and contents.
- **Feature Engineering**: Create new features from the existing data to enhance model performance.
- **Data Visualization**: Visualize the data to understand trends and relationships between variables.
- **Technical Indicators**: Calculate various technical indicators such as SMA, Bollinger Bands, MACD, RSI, and standard deviation.
- **Normalization**: Normalize the data to prepare it for modeling.
- **Train-Test Split**: Split the data into training and testing sets using time series cross-validation.
- **Model Training**: Train machine learning models and evaluate their performance.
- **Validation**: Validate the results and visualize predictions versus actual values.

## Installation

1. Clone the repository or download the source code.
2. Ensure you have Python 3.6+ installed.
3. Install the required libraries using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Place your dataset in the appropriate directory.
2. Run the main script to perform data analysis and model training:
    ```bash
    python main.py
    ```

## Dataset

The dataset used in this project is a CSV file containing historical gold prices and other financial indicators. Ensure the CSV file is placed in the `/kaggle/input/gold-price-prediction-dataset/` directory.

## Features

- **Adjusted Close Prices**: Prices of various financial indices.
- **Daily Returns**: Computed daily returns of all features.
- **Technical Indicators**: SMA, Bollinger Bands, MACD, RSI, and standard deviation.

## Visualization

The project includes various visualizations such as:
- Effect of index prices on gold rates.
- Daily returns of all features.
- Scatter plots of features against gold prices.
- Correlation matrix of features.
- Technical indicator plots.

## Model Validation

The project uses the `validate_result` function to evaluate model performance. The function computes RMSE and R2 scores and plots predicted versus actual values.

## Technical Indicators Functions

- `calculate_SMA(df, periods=15)`
- `calculate_BB(df, periods=15)`
- `calculate_MACD(df, nslow=26, nfast=12)`
- `calculate_RSI(df, periods=14)`
- `calculate_stdev(df, periods=5)`

