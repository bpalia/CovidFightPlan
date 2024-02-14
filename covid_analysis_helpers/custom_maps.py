import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import List

my_colors = ["red"]
mapbox_style = "carto-positron"


def map_dots(
    df: pd.DataFrame,
    lat: str,
    lon: str,
    size: str,
    hover_name: str,
    labels: List[str],
    title: str,
    size_max: int = 40,
    color: str = None,
    color_seq: List[str] = ["red"],
    animation: str = None,
) -> Figure:
    """Function to draw information on map."""
    fig = px.scatter_mapbox(
        df,
        lat=lat,
        lon=lon,
        size=size,
        size_max=size_max,
        color=color,
        color_discrete_sequence=color_seq,
        opacity=0.5,
        zoom=6,
        hover_name=hover_name,
        hover_data={
            size: False,
            lat: False,
            lon: False,
            color: False,
        }
        | {i: True for i in labels},
        labels={i: i.split("_")[0].title() for i in labels},
        title=title,
        animation_frame=animation,
    )
    fig.update_layout(mapbox_style=mapbox_style, width=600, height=600)
    return fig
