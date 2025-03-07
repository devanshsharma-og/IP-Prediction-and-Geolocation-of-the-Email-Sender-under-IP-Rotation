import random
import numpy as np
from datetime import datetime
from email.message import EmailMessage
import re

# Function to generate a random IP address
def random_ip():
    return '.'.join(str(random.randint(0, 255)) for _ in range(4))

# Function to simulate IP rotation and generate training data
def simulate_ip_rotation(num_samples, num_senders, rotation_pool):
    X_train = []
    y_train = []
    for _ in range(num_samples):
        sender_ip_index = random.randint(0, num_senders - 1)
        rotated_ips = [random_ip() for _ in range(rotation_pool)]
        for ip in rotated_ips:
            sample = {
                'source_ip_freq': np.random.rand(1, num_senders),
                'time_between_emails': np.random.rand(100, 1),
                'timestamp': np.random.rand(100, 1),
                'time_discrepancies': np.random.rand(100, 1),
                'x_originating_ip': np.random.randint(0, 2, 256)
            }
            X_train.append(sample)
            y_train.append(sender_ip_index)
    return X_train, y_train

# Function to generate a random email with IPs in Received headers
def generate_random_email_header():
    received_headers = []
    for _ in range(random.randint(2, 5)):
        received_headers.append(f"from {random_ip()} by {random_ip()} with ESMTP id {random.randint(100000, 999999)}; {datetime.now()}")

    x_originating_ip = random_ip()

    email_message = EmailMessage()
    email_message['From'] = 'sender@example.com'
    email_message['To'] = 'receiver@example.com'
    email_message['Subject'] = 'Test email'
    email_message['X-Originating-IP'] = f"[{x_originating_ip}]"
    for received in received_headers:
        email_message['Received'] = received

    return email_message.as_bytes()
