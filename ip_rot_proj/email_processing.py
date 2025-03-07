import re
import numpy as np
from email.parser import BytesParser
from email import policy
import ipaddress

# Function to extract features from email header
def extract_features_from_email(email_bytes, sender_ips, num_ips, sequence_length):
    email_message = BytesParser(policy=policy.default).parsebytes(email_bytes)
    received_headers = email_message.get_all('Received')
    
    ip_pattern = re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')
    ips = []
    
    # Extract IPs from 'Received' headers
    for header in received_headers:
        ips.extend(ip_pattern.findall(header))
    
    source_ip_freq = np.zeros((1, len(sender_ips)))  # Initialize frequency array for sender IPs
    
    # Count the frequency of each IP in the 'Received' headers
    for ip in ips:
        if ip in sender_ips:
            index = sender_ips.index(ip)
            source_ip_freq[0][index] += 1
    
    # Placeholder random features (you should replace these with actual logic)
    time_between_emails = np.random.rand(sequence_length, 1)
    timestamp = np.random.rand(sequence_length, 1)
    time_discrepancies = np.random.rand(sequence_length, 1)
    
    # Extract the 'X-Originating-IP' field if present
    x_originating_ip = email_message.get('X-Originating-IP')
    x_originating_ip_array = np.zeros((num_ips,))
    
    if x_originating_ip:
        x_originating_ip = re.findall(ip_pattern, x_originating_ip)
        if x_originating_ip:
            # Convert IP to an integer index for `x_originating_ip_array`
            ip_int = int(ipaddress.IPv4Address(x_originating_ip[0]))
            x_originating_ip_array[ip_int % num_ips] = 1
    
    # Return the features as a dictionary
    return {
        'source_ip_freq': source_ip_freq,
        'time_between_emails': time_between_emails,
        'timestamp': timestamp,
        'time_discrepancies': time_discrepancies,
        'x_originating_ip': x_originating_ip_array
    }
