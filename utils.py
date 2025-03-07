import numpy as np
import requests
import ipaddress
from email_processing import extract_features_from_email  # Make sure the correct import for email feature extraction

# Function to get geolocation of an IP address
def get_geolocation(ip_address):
    url = f"http://ipinfo.io/{ip_address}/json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        country = data.get('country', 'Unknown Country')
        region = data.get('region', 'Unknown Region')
        city = data.get('city', 'Unknown City')
        loc = data.get('loc', 'Unknown Location')
        return f"{city}, {region}, {country}", loc
    else:
        return None, None

# Function to map sender IP index to the actual IP address
def get_true_ip(predicted_sender_ip_index, sender_ips):
    return [sender_ips[index] for index in predicted_sender_ip_index]

# Function to predict the sender's IP based on email features
def predict_sender_ip(model, email_bytes, sender_ips, num_ips, sequence_length):
    features = extract_features_from_email(email_bytes, sender_ips, num_ips, sequence_length)
    
    features_array = {
        'source_ip_freq': np.array([features['source_ip_freq']]),
        'time_between_emails': np.array([features['time_between_emails']]),
        'timestamp': np.array([features['timestamp']]),
        'time_discrepancies': np.array([features['time_discrepancies']]),
        'x_originating_ip': np.array([features['x_originating_ip']])
    }

    predictions = model.predict(features_array)
    predicted_sender_ip_index = np.argmax(predictions, axis=1)
    return predicted_sender_ip_index
