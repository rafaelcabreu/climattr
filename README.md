
# ClimAttr

ClimAttr is a Python package providing a comprehensive toolbox for extreme event climate attribution. It leverages powerful libraries such as xarray, pandas, geopandas, and scipy to analyze climate data, generate statistical metrics, and visualize results.

## Features

-  **Attribution Metrics Calculation**:

- Calculate Probability Ratio (PR), Fraction of Attributable Risk (FAR), and Return Periods (RP) with confidence intervals.

- Supports both additive and multiplicative correction methods.

-  **Data Handling**:

- Efficiently handle and process NetCDF files from CMIP6 and other climate datasets.

- Mask and filter data based on geographical boundaries using shapefiles or bounding boxes.

**Visualization Tools**:

- Create histograms, quantile-quantile (QQ) plots, and return period plots to visualize climate data.

- Supports custom geographical plotting with features like country borders, state lines, and shapefile overlays.

  

## Installation

  

To install ClimAttr, you can install directly from GitHub using pip:

Alternatively, you can install directly from GitHub using pip:

```bash
pip  install  git+https://github.com/rafaelcabreu/climattr.git
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
metrics =  eea.attribution.attribution_metrics(
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
filtered_data =  eea.spatial.filter_area(
	dataset, mask='path_to_shapefile.shp', plot_area=True
)

# Filter data based on a bounding box
filtered_data_box =  eea.spatial.filter_area(
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


