# app.py
from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('Abhinaya1_lastfinal.csv')

# Define features and target
X = df.drop(columns=['ProcessName', 'CPUUsage', 'Timestamp'])
y = df['CPUUsage']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the machine learning model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Define a function to predict CPU usage
def predict_cpu_usage(input_data):
    input_data_scaled = scaler.transform([input_data])
    prediction = model.predict(input_data_scaled)
    return prediction[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    process_name = request.form['process_name']
    memory_usage = float(request.form['memory_usage'])
    disk_write_bytes = float(request.form['disk_write_bytes'])
    network_sent_bytes = float(request.form['network_sent_bytes'])
    system_cpu_usage = float(request.form['system_cpu_usage'])
    system_memory_usage = float(request.form['system_memory_usage'])
    system_disk_write_bytes = float(request.form['system_disk_write_bytes'])
    system_network_sent_bytes = float(request.form['system_network_sent_bytes'])

    input_data = [memory_usage, disk_write_bytes, network_sent_bytes, system_cpu_usage,
                  system_memory_usage, system_disk_write_bytes, system_network_sent_bytes]
    prediction = predict_cpu_usage(input_data)
    return render_template('result.html', process_name=process_name, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
