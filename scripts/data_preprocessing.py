import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def handle_missing_values(df):
    """
    Handle missing values in the dataset.
    Impute missing values where necessary, or drop rows/columns with too many missing values.
    """
    # Check for missing values
    missing = df.isnull().sum()

    # Drop rows with missing values
    df_cleaned = df.dropna()

    return df_cleaned

def remove_duplicates(df):
    """Remove duplicate rows from the dataframe."""
    return df.drop_duplicates()

def correct_data_types(df):
    """Correct data types, especially for datetime fields."""
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    return df


def ip_to_int(ip):
    """
    Convert an IP address to an integer. Handles if the IP address is a floating-point number.
    """
    if isinstance(ip, float):
        ip = int(ip)  # Convert float to integer if needed
    return int(ip)

def map_ip_to_country(fraud_data, ip_data):
    """
    Map the IP address to a country using a range from the IP-to-Country dataset.
    """
    fraud_data['ip_address'] = fraud_data['ip_address'].apply(ip_to_int)

    ip_data['lower_bound_ip_address'] = ip_data['lower_bound_ip_address'].astype(int)
    ip_data['upper_bound_ip_address'] = ip_data['upper_bound_ip_address'].astype(int)

    def get_country(ip):
        match = ip_data[(ip_data['lower_bound_ip_address'] <= ip) & (ip_data['upper_bound_ip_address'] >= ip)]
        return match['country'].values[0] if not match.empty else 'Unknown'

    fraud_data['country'] = fraud_data['ip_address'].apply(get_country)
    return fraud_data



def create_transaction_frequency(df):
    """Create transaction frequency and velocity features."""
    df['transaction_count'] = df.groupby('user_id')['user_id'].transform('count')
    df['time_diff'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
    df['transaction_velocity'] = df['purchase_value'] / df['time_diff']
    return df

def create_time_features(df):
    """Create hour_of_day and day_of_week features from the purchase time."""
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    return df



def normalize_and_scale(df):
    """Normalize numerical features and encode categorical features."""
    
    # Scaling numerical columns
    scaler = StandardScaler()
    df[['purchase_value', 'transaction_velocity']] = scaler.fit_transform(df[['purchase_value', 'transaction_velocity']])

    # Encoding categorical columns
    encoder = LabelEncoder()
    df['source'] = encoder.fit_transform(df['source'])
    df['browser'] = encoder.fit_transform(df['browser'])
    df['sex'] = encoder.fit_transform(df['sex'])

    return df
