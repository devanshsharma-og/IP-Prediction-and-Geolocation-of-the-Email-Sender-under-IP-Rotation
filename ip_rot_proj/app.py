import streamlit as st
import random
import numpy as np
import tensorflow as tf
import folium
import pandas as pd
import matplotlib.pyplot as plt
from model import build_model
from data_generation import simulate_ip_rotation, generate_random_email_header
from email_processing import extract_features_from_email
from utils import get_geolocation, get_true_ip, predict_sender_ip
from ip_utils import sender_ips, num_sender_ips, sequence_length, num_ips

# Parameters
rotation_pool = 10  # Pool of rotated IPs for each sender

# Streamlit UI
st.set_page_config(page_title="IP Prediction and Geolocation", layout="wide")
st.title("IP Prediction and Geolocation of the Email Sender under IP Rotation")
st.sidebar.header("Control Panel")

# Load or build the model
if 'model' not in st.session_state:
    # Pass the required arguments to build_model function
    st.session_state.model = build_model(num_sender_ips, sequence_length, num_ips)

# Initialize training data
if 'X_train_data' not in st.session_state:
    st.session_state.X_train_data = []
if 'y_train_data' not in st.session_state:
    st.session_state.y_train_data = []

# Input section for training
st.sidebar.subheader("Model Training Configuration")
num_samples = st.sidebar.number_input("Number of samples to generate", min_value=1, max_value=500, value=10)
epochs = st.sidebar.number_input("Number of epochs", min_value=1, max_value=50, value=5)
batch_size = st.sidebar.number_input("Batch size", min_value=1, max_value=64, value=32)

if st.sidebar.button("Generate and Train Model"):
    # Clear previous training data before generating new data
    st.session_state.X_train_data.clear()
    st.session_state.y_train_data.clear()

    X_train_data, y_train_data = simulate_ip_rotation(num_samples, num_sender_ips, rotation_pool)
    y_train_data = tf.keras.utils.to_categorical(y_train_data, num_classes=num_sender_ips)

    # Prepare the data for model training
    X_train = {
        'source_ip_freq': np.array([x['source_ip_freq'] for x in X_train_data]),
        'time_between_emails': np.array([x['time_between_emails'] for x in X_train_data]),
        'timestamp': np.array([x['timestamp'] for x in X_train_data]),
        'time_discrepancies': np.array([x['time_discrepancies'] for x in X_train_data]),
        'x_originating_ip': np.array([x['x_originating_ip'] for x in X_train_data])
    }

    st.session_state.X_train_data = X_train
    st.session_state.y_train_data = y_train_data

    # Train the model
    st.session_state.model.fit(X_train, y_train_data, epochs=epochs, batch_size=batch_size)
    st.success("Training complete!")

# Training Data Samples section
st.sidebar.subheader("Training Data Samples")
if st.sidebar.button("Show Training Data Samples"):
    if st.session_state.X_train_data:
        samples_df = pd.DataFrame({
            'source_ip_freq': [x.flatten() for x in st.session_state.X_train_data['source_ip_freq']],
            'time_between_emails': [x.flatten() for x in st.session_state.X_train_data['time_between_emails']],
            'timestamp': [x.flatten() for x in st.session_state.X_train_data['timestamp']],
            'time_discrepancies': [x.flatten() for x in st.session_state.X_train_data['time_discrepancies']],
            'x_originating_ip': [x for x in st.session_state.X_train_data['x_originating_ip']]
        })
        samples_df = samples_df.head(100)  # Limit to 100 samples
        st.dataframe(samples_df)
    else:
        st.warning("No training data available. Please generate training data first.")

# Email Header Generation section
st.sidebar.subheader("Email Header Generation")
num_headers = st.sidebar.number_input("Number of Sample Email Headers to Generate", min_value=1, max_value=50, value=20)
if st.sidebar.button("Show Sample Email Headers"):
    sample_headers = [generate_random_email_header() for _ in range(num_headers)]
    st.write("Generated Sample Email Headers:")
    for header in sample_headers:
        st.text(header.decode())

# Prediction section
st.sidebar.subheader("IP Prediction")
if st.sidebar.button("Predict Sender's IP"):
    random_email_bytes = generate_random_email_header()
    # Pass the required arguments to predict_sender_ip function
    predicted_sender_ip_index = predict_sender_ip(st.session_state.model, random_email_bytes, sender_ips, num_ips, sequence_length)
    true_ip_addresses = get_true_ip(predicted_sender_ip_index, sender_ips)

    # Retrieve geolocation for the predicted IP addresses
    for ip_address in true_ip_addresses:
        geolocation, loc = get_geolocation(ip_address)
        if loc:
            try:
                # Attempt to unpack the geolocation into latitude and longitude
                latitude, longitude = loc.split(',')
                st.write(f"Predicted IP: {ip_address}, Geolocation: {geolocation}")

                # Create a map
                map = folium.Map(location=[float(latitude), float(longitude)], zoom_start=10)
                folium.Marker([float(latitude), float(longitude)], popup=geolocation).add_to(map)

                # Render the map in Streamlit
                st.subheader("Location on Map")
                map_html = map._repr_html_()  # Get the HTML representation
                st.components.v1.html(map_html, height=500)
            except ValueError:
                # Handle case where geolocation string is not in the expected format
                st.write(f"Predicted IP: {ip_address}, Geolocation: Invalid format")
        else:
            st.write(f"Predicted IP: {ip_address}, Geolocation: Not found")

# Data Visualization section
st.sidebar.header("Visualization")
if st.sidebar.button("Show Model Inputs/Outputs Visualization"):
    fig, ax = plt.subplots()
    if 'y_train_data' in st.session_state:
        counts = [np.sum(st.session_state.y_train_data[:, i]) for i in range(num_sender_ips)]
        ax.bar(range(num_sender_ips), counts, label='Counts per Sender IP', color='skyblue')
        ax.set_xticks(range(num_sender_ips))
        ax.set_xticklabels(sender_ips)
        ax.set_xlabel('Sender IP')
        ax.set_ylabel('Counts')
        ax.set_title('Counts of Predicted Sender IPs')
        st.pyplot(fig)
    else:
        st.warning("No training data available for visualization.")

    # Pie chart visualization
    st.subheader("Pie Chart of Sender IPs")
    if counts:
        fig, ax = plt.subplots()
        ax.pie(counts, labels=sender_ips, autopct='%1.1f%%', startangle=90)
        ax.set_title('Distribution of Sender IPs')
        st.pyplot(fig)

    # Time series graph for email generation
    st.subheader("Email Generation Time Series")
    if 'X_train_data' in st.session_state and len(st.session_state.X_train_data['timestamp']) > 0:
        time_series_data = np.array([np.sum(x) for x in st.session_state.X_train_data['timestamp']])
        plt.figure(figsize=(10, 5))
        plt.plot(time_series_data, label='Email Generation Over Time', color='blue')
        plt.xlabel('Sample Index')
        plt.ylabel('Email Generation Count')
        plt.title('Time Series of Email Generation')
        plt.legend()
        st.pyplot(plt)
