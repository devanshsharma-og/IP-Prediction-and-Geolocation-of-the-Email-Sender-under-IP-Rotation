# IP-Prediction-and-Geolocation-of-the-Email-Sender-under-IP-Rotation
This application predicts the IP address of an email sender based on the provided email headers using a machine learning model built with TensorFlow. It also provides geolocation data for the predicted IP address and visualizations of the model's performance.

## Features

- Simulates IP rotation for sender email addresses.
- Trains a neural network model to predict the sender's IP.
- Displays geolocation of the predicted IP using IP address lookup.
- Visualizes model performance and generated training data.

## Setup Instructions

Follow the instructions below to set up, run, and deploy the app.

### 1. Clone the Repository

Clone this repository to your local machine by running:

```bash
git clone https://github.com/devanshsharma-og/IP-Prediction-and-Geolocation-of-the-Email-Sender-under-IP-Rotation.git
cd IP-Prediction-and-Geolocation-of-the-Email-Sender-under-IP-Rotation

pip install -r requirements.txt


streamlit run app.py


6.6 Final Testing and Validation
Before finalizing the deployment, end-to-end testing was performed using test datasets and synthetic email headers. Metrics such as prediction accuracy, latency, and geolocation accuracy were evaluated. The results demonstrated that the system met the project objectives effectively.
This comprehensive testing and integration ensured that the system was production-ready, offering a robust solution for predicting email sender IPs under rotation and providing meaningful geolocation insights.
