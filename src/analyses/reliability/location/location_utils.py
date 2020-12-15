from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
from matplotlib import axes
from shapely.geometry import Point, Polygon

REPO_DIR = Path(__file__).parents[2]

# Define latitude and longitude of India
LATITUDE_MIN, LATITUDE_MAX = 5, 37
LONGITUDE_MIN, LONGITUDE_MAX = 67, 90

LONGITUDE_SPLIT1 = 75
LONGITUDE_SPLIT2 = 79


def make_polygon(latitude_min: float, latitude_max: float, longitude_min: float, longitude_max: float) -> Polygon:
    return Polygon([
        (longitude_min, latitude_min), (longitude_min, latitude_max), (longitude_max, latitude_max),
        (longitude_max, latitude_min), (longitude_min, latitude_min)])


def draw_map(df: pd.DataFrame) -> axes._base._AxesBase:
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    polygon = Polygon([
        (LONGITUDE_MIN, LATITUDE_MIN), (LONGITUDE_MIN, LATITUDE_MAX), (LONGITUDE_MAX, LATITUDE_MAX),
        (LONGITUDE_MAX, LATITUDE_MIN), (LONGITUDE_MIN, LATITUDE_MIN)])
    world = geopandas.clip(world, polygon)

    # Create world shape
    ax = world.plot(figsize=(15, 10))

    # Place points on the map
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf = geopandas.GeoDataFrame(df, geometry=geometry)
    gdf.plot(ax=ax, marker='o', color='red', markersize=15)
    return ax


def draw_polygon(ax: axes._base._AxesBase, polygon: Polygon):
    poly_gdf = geopandas.GeoDataFrame([1], geometry=[polygon])
    poly_gdf.boundary.plot(ax=ax, color="red")


def draw_distributions(series: List[pd.Series], labels: List[str], caption: str = ""):
    plt.figure(figsize=(20, 7))
    bins = np.linspace(0, 20, 50)
    for s, label in zip(series, labels):
        plt.hist(s, bins, alpha=0.5, label=label)
    plt.legend(loc='upper right')
    plt.title('Distribution of error for 2 different locations. ' + caption)
    plt.xlabel('error')
    plt.ylabel('occurance count')
    plt.show()


def split_by_location(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_left = df[df["longitude"] < LONGITUDE_SPLIT1]
    df_middle = df[(LONGITUDE_SPLIT1 <= df["longitude"]) & (df["longitude"] <= LONGITUDE_SPLIT2)]
    df_right = df[df["longitude"] > LONGITUDE_SPLIT2]
    return df_left, df_middle, df_right
