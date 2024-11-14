# Fan Coil Unit Simulation Framework and Isolation Forest + PCA Anomaly Detection

## Description:

This repo contains the simulation framework for the Fan Coil Unit (FCU) and the anomaly detection model using Isolation Forest and PCA. The FCU simulation framework is built using random sampling techniques to generate simulated data over the course of one year. The anomaly detection model is built using the Isolation Forest and PCA algorithms to detect anomalies in the FCU data. The model is trained on the simulated data and tested on the real FCU data.

## Files:
- `FCU` folder: Contains the `.csv` files for the simulated FCU data.
- `features` folder: Contains relevant mappings for Fault codes, feature descriptions and table names. 
- `main.py`: The main file that runs the simulation framework and anomaly detection model and saves the results in the `models/data` folder. 


### Getting Started: 
- Install the `requirements.txt` file using the command line with: `pip install -r requirements.txt`
- Ensure that the `.csv` files from the URL below are located in the `FCU` folder: <br> https://faultdetection.lbl.gov/dataset/simulated-fan-coil-unit-data-set/.

### Running the Simulation Framework and Anomaly Detection Model:
- Run the `main.py` file using the command line with: `python main.py`
  - Specify the relevant path for the `.csv` files, and the `features/Faults.csv` file in the `main.py` file.
  - The results will be saved in the `models/data` folder as `.gzip` files.
- Run the `all_models.ipynb` notebook to run the Isolation Forest and PCA models on the simulated FCU data.
  - Isolation Forest is applied to both Raw and PCA-decomposed data, results are stored in the `models/reg` and `models/pca` folders respectively.
  - The Decomposed dataset is also stored within the `models/decomp` folder. 
  - Results from the models are stored in the `models/results` folder. 

### Evaluating Results and Visualizing Anomalies:
- Run the `model_eval.ipynb` notebook to view the results of the anomaly detection model and visualize the anomalies in the FCU data.
- Run the `fault_analysis.ipynb` notebook to view the Accuracy and AUC scores from each model 
