# Overview

| File      | Description                                                                 |
|-----------|-----------------------------------------------------------------------------|
| `main.py` | Entry point. Sets up the training environment, loads modules, and starts training. |
| `model.py`| Loads the model to predict on test data and computes errors vs. ground truth.     |
| `Importdata.py`| Load data     |


# Install requirements

 **PyTorch**, **NumPy**, **scikit-learn**, and **haversine**.  

## Usage

To run the project, follow these steps:

1. **Install dependencies**  
   Make sure the required packages are installed.

2. **Configure paths**  
   Edit `Importdata.py` to set the correct paths for your data and model directories.

3. **Start training**  
   Run the training script:
   ```bash
   python main.py
## Data

- **Air Quality Dataset **  
  National Urban Air Quality Real-Time Publishing Platform: <https://air.cnemc.cn/>

- **Meteorological Data (ERA5, hourly single levels)**  
  Copernicus Climate Data Store: <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels>


