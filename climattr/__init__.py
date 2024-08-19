from climattr.attribution import (
    attribution_metrics,
    histogram_plot,
    rp_plot
)
from climattr.exploratory import (
    timeseries_plot
)
from climattr.main import ClimAttr
from climattr.spatial import (
    _mask_area,
    _plot_area,
    filter_area
)

# attribution methods
ClimAttr.attribution_metrics = attribution_metrics
ClimAttr.histogram_plot = histogram_plot
ClimAttr.rp_plot = rp_plot

# spatial methods
ClimAttr._mask_area = _mask_area
ClimAttr._plot_area = _plot_area
ClimAttr.filter_area = filter_area

# exploratory methods
ClimAttr.timeseries_plot = timeseries_plot