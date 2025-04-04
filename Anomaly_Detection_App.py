import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.preprocessing import StandardScaler

# Load the trained model
model_path = 'anomaly_detection_model.pkl'
model = joblib.load(model_path)

# Load the label encoder
label_encoder_path = 'label_encoders.pkl'
label_encoders = joblib.load(label_encoder_path)

# Function to preprocess input data
def preprocess_input(data):
    # Numerical features to be scaled
    numerical_features = [
        "Bwd Packet Length Std", "Flow Bytes/s", "Total Length of Fwd Packets",
        "Fwd Packet Length Std", "Flow IAT Std", "Flow IAT Min",
        "Fwd IAT Total", "Flow Duration", "Flow IAT Mean",
        "Flow IAT Max", "Total Length of Bwd Packets", "Bwd Packet Length Max",
        "Fwd Packet Length Mean", "Fwd Packet Length Min", "Flow Packets/s",
        "Bwd Packet Length Mean", "Total Backward Packets", "Fwd Packet Length Max",
        "Total Fwd Packets", "Bwd Packet Length Min"
    ]

    # String features that need special handling
    string_features = ["Flow Bytes/s", "Flow Packets/s"]

    # Handle non-numeric values in the string features
    for feature in string_features:
        # Replace 'Infinity' with -1 and 'NaN' with 0
        data[feature] = data[feature].replace('Infinity', -1)
        data[feature] = data[feature].replace('NaN', 0)

        # Convert non-numeric values to 0
        number_or_not = []
        for value in data[feature]:
            try:
                k = int(float(value))  # Try to convert to float, then int
                number_or_not.append(int(k))
            except:
                number_or_not.append(0)  # If conversion fails, append 0
        data[feature] = number_or_not

    # Scale numerical features
    scaler = StandardScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    return data

# Streamlit app
st.title("Anomaly Detection App")

st.write("""
This app uses a pre-trained XGBoost model to detect anomalies in
network traffic data.
Upload a CSV file.
""")

# File uploader
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded file
    input_data = pd.read_csv(uploaded_file)

    

    # Preprocess the input data
    processed_data = preprocess_input(input_data)

    # Make predictions
    predictions = model.predict(processed_data)

    # Decode the predictions using the label encoder
    decoded_predictions = label_encoders['Label'].inverse_transform(predictions)

    # Add predictions to the dataframe
    input_data['Detected_Label'] = decoded_predictions

    # # Display the predictions
    # st.write("Predictions:")
    # st.write(input_data)

    # Count the occurrences of each label
    label_counts = pd.Series(decoded_predictions).value_counts().reset_index()
    label_counts.columns = ["Label", "Count"]

    

    # Create an interactive bar chart using Plotly
    fig = px.bar(
        label_counts,
        x="Label",
        y="Count",
        text="Count",  # Display count on top of bars
        title="Distribution of Detected Attack Types",
        labels={"Label": "Attack Type", "Count": "Number of Detection"},
    )

    # Customize hover text
    fig.update_traces(
        hovertemplate="<b>Attack Type:</b> %{x}<br><b>Count:</b>%{y}<extra></extra>"
    )

    # Rotate x-axis labels for better readability
    fig.update_layout(xaxis_tickangle=-45)

    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Display the counts in a table format
    st.write("Detected Anomalies Count")
    st.write(label_counts)

    # Display the uploaded data
    st.write("Detected Anomalies Dataset:")
    st.write(input_data)

    # Optionally, save the results to a CSV file
    if st.button("Save to CSV"):
        input_data.to_csv('detected_anomalies.csv', index=False)
        st.success("Data saved to detected_anomalies.csv")
else:
    st.write("Please upload a CSV file.")