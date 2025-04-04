# Anomaly Detection in Network Traffic Using Machine Learning in IoT Devices

## Overview
This project focuses on detecting anomalies in network traffic using machine learning techniques (using XGBoost Classifier). The goal is to identify suspicious activities that could indicate cyber threats, such as unauthorized access, DDoS attacks, or malware infections.

## Features
- **Data Preprocessing:** Cleaning and preparing network traffic data for analysis.
- **Feature Engineering:** Extracting and selecting relevant features to improve model performance.
- **Machine Learning Models:** Implementing and evaluating various algorithms for anomaly detection.
- **Performance Evaluation:** Measuring accuracy, precision, recall, and F1-score to assess model effectiveness.
- **Visualization:** Graphical representation of normal and anomalous network behavior.

## Dataset
The project utilizes publicly available network traffic datasets, such as **UNSW-NB15**, **CICIDS2017**, or **KDD Cup 99**, to train and test the models.

## Technologies Used
- Python
- Scikit-learn
- XG-Boost
- Pandas & NumPy
- Matplotlib & Seaborn

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/thomas-vargheese/Network-Anomaly-Detection.git
   ```
2. Navigate to the project directory:
   ```sh
   cd anomaly-detection
   ```

## Usage
1. Clone the repository or download all the files and place it in a single folder(files : ```Anomaly_Detection_App.py```,```anomaly_detection_model.pkl```,``` label_encoders.pkl```).
2. Make sure you have installed the Streamlit framework of python on your machine.
3.    To install Streamlit, open a terminal, ensure you have Python and pip installed, and then run the command
   ```sh
   pip install streamlit
   ```
   You can then verify the installation by running
   ```sh
   streamlit hello
   ```
4. Open Command prompt ```cmd``` on your Windows machine and run ``` sh streamlit run Anomaly_Detection_App.py```.
5. Upload the dataset to the dashboard.
6. Result will be displayed a chart and further analysis can be done. 


## Results
The model successfully detects anomalous traffic patterns with high accuracy. Performance metrics and visualizations are provided in the `Final_XG_Boost.ipynb` file.


## Acknowledgments
Special thanks to researchers and contributors in the field of network security and machine learning.



