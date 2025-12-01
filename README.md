  

# ClimAttr

  

ClimAttr is a Python package providing a comprehensive toolbox for extreme event climate attribution. It leverages powerful libraries such as xarray, pandas, geopandas, and scipy to analyze climate data, generate statistical metrics, and visualize results.

## Instalation

You can install **climattr** using `pip` directly from github:

```
pip install git+https://github.com/rafaelcabreu/climattr
```
  
## Features

**Attribution Metrics Calculation**:

- Calculate Probability Ratio (PR), Fraction of Attributable Risk (FAR), and Return Periods (RP) with confidence intervals.

- Supports both additive and multiplicative correction methods.

- Supportts the World [Weather Attribution (WWA)](https://ascmo.copernicus.org/articles/6/177/2020/) method using GEV, Normal, GPD, or Gamma distributions and the linear and exponential approaches.

**Data Handling**:

- Efficiently handle and process NetCDF files from CMIP6 and other climate datasets.

- Mask and filter data based on geographical boundaries using shapefiles or bounding boxes.

- This packages leaverages on [xclim](https://xclim.readthedocs.io/en/stable/api_indicators.html) to calculate indices such as Maximum 1 to N days Rainfall, that are usual in attribution studies. However it can also be used to calculate other relevant indices such as FWI, SPI, and SPEI which might be useful depending on the application. 

**Visualization Tools**:

- Create histograms, quantile-quantile (QQ) plots, and return period plots to visualize climate data.

- Supports custom geographical plotting with features like country borders, state lines, and shapefile overlays.

**Impact Help Functions**:
 - A set of functions used to handle geolocalized dataframes using `pandas` and `geopandas` that might help doing impact analysis, such as geolocating the impact dataframe, resampling it spatially and temporally.

  

## Installation

  
To install ClimAttr, you can install directly from GitHub using pip:

Alternatively, you can install directly from GitHub using pip:

```bash
pip  install  git+https://github.com/rafaelcabreu/climattr@master
```

```

## Usage

  
### 1. Calculating Attribution Metrics

 
```python
import xarray as xr
import climattr as eea

# Load your datasets
all_data = xr.open_dataset('path_to_all_scenario.nc')
nat_data = xr.open_dataset('path_to_nat_scenario.nc')

# Calculate attribution metrics
metrics = eea.attribution.attribution_metrics(
	all=all_data, nat=nat_data, fit_function=some_fit_function, thresh=threshold_value
)

print(metrics)
```

### 2. Masking and Filtering Data

 
```python
import xarray as xr
import climattr as eea

# Load your dataset
dataset = xr.open_dataset('path_to_your_dataset.nc')

# Apply a mask using a shapefile
filtered_data = eea.spatial.filter_area(
	dataset, mask='path_to_shapefile.shp', plot_area=True
)

# Filter data based on a bounding box
filtered_data_box = eea.spatial.filter_area(
	dataset, box=[xmin, xmax, ymin, ymax], plot_area=True
)
```

### 3. Visualization

#### a. Validation Histogram Plot

```python
import xarray as xr
import matplotlib.pyplot as plt
import climattr as eea

# Load your dataset
obs_data = xr.open_dataset('path_to_your_dataset.nc')
model_data = xr.open_dataset('path_to_your_dataset.nc')

fig, ax = plt.subplots()

eea.validation.histogram_plot(
	ax, obs=obs_data, all=model_data, fit_function=some_fit_function
)

plt.show()
```

#### b. Validation Quantile-Quantile (QQ) Plot


```python
import xarray as xr
import matplotlib.pyplot as plt
import climattr as eea

# Load your dataset
obs_data = xr.open_dataset('path_to_your_dataset.nc')
model_data = xr.open_dataset('path_to_your_dataset.nc')

fig, ax = plt.subplots()

eea.validation.qq_plot(ax, obs=obs_data, all=model_data)

plt.show()
```

## Notebooks

We provide a set of Jupyter notebooks that demonstrate how to use **climattr** for various climate attribution tasks. These notebooks serve as practical guides, showing step-by-step processes and visualizations.

### Available Notebooks

1.  [Attribution Package](https://github.com/rafaelcabreu/climattr/blob/master/notebooks/AttributionPackageExample.ipynb): An overview of the package's functionalities and how to get started.
2.  [Attribution CLI](https://github.com/rafaelcabreu/climattr/blob/master/notebooks/AttributionCLIExample.ipynb): Example of how to use the CLI functions for people that are not familiar with python or don't want to use the package.
3.  [WWA Attribution](https://github.com/rafaelcabreu/climattr/blob/master/notebooks/AttributionPackageWWA.ipynb): Step-by-step implementation of the World Weather Attribution method.
4.  [Impact Help Functions](https://github.com/rafaelcabreu/climattr/blob/master/notebooks/ImpactPackageExample.ipynb): A set of functions used to handle geolocalized dataframes using `pandas` and `geopandas` that might help doing impact analysis.

### How to Use the Notebooks

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/rafaelcabreu/climattr.git
    ```
2. **Navigate to the Notebooks Directory**:
	```bash
	cd climattr/notebooks
	```
3. **Install Required Dependencies**:
Ensure all dependencies are installed, possibly using a virtual environment
   ```bash
   pip install -r requirements.txt
   ```
 4. **Launch Jupyter Notebook**:
	 ```bash
	 jupyter notebook
	 ``` 
4. **Open and Run Notebooks**:
Select a notebook to open and follow the instructions within.
