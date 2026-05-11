"""Build external-data features for the NYC Airbnb dataset.

Inputs (under data/external/):
  - tl_2019_36_tract/tl_2019_36_tract.shp  (TIGER 2019 census tracts, NY state)
  - acs2019_b19013_nytracts.json           (ACS 2015-2019 5-yr median HH income, NYC tracts)
  - mta_subway_stations.csv                (MTA station list, lat/lon)

Output:
  - data/derived/listings_external.parquet (one row per listing id, with new features)

Features added per listing:
  - tract_geoid:         11-char census tract GEOID
  - tract_median_income: ACS 2015-2019 5-yr median household income (USD)
  - dist_subway_km:      haversine distance to nearest MTA subway station
  - dist_central_park_km, dist_jfk_km, dist_lga_km, dist_wall_st_km:
                         haversine distances to additional NYC anchors
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / 'data' / 'external'
DERIVED = ROOT / 'data' / 'derived'
LISTINGS_CSV = ROOT / 'data' / 'listings.csv'

NYC_COUNTY_FIPS = {'005', '047', '061', '081', '085'}

# ACS uses extreme negative integers as sentinels for "not available", "below
# range", "estimate not reported", etc. Any value <= 0 is non-meaningful for
# median household income.
ACS_SENTINEL_THRESHOLD = 0

ANCHORS_LL = {
    'central_park': (40.7829, -73.9654),
    'jfk':          (40.6413, -73.7781),
    'lga':          (40.7769, -73.8740),
    'wall_st':      (40.7069, -74.0090),
}


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_listings() -> pd.DataFrame:
    df = pd.read_csv(LISTINGS_CSV)
    df = df[df['price'] > 0].reset_index(drop=True)
    return df


def load_acs_income() -> pd.DataFrame:
    raw = json.load(open(EXT / 'acs2019_b19013_nytracts.json'))
    acs = pd.DataFrame(raw[1:], columns=raw[0])
    acs['B19013_001E'] = pd.to_numeric(acs['B19013_001E'], errors='coerce')
    acs.loc[acs['B19013_001E'] <= ACS_SENTINEL_THRESHOLD, 'B19013_001E'] = np.nan
    acs['GEOID'] = acs['state'] + acs['county'] + acs['tract']
    return acs.rename(columns={'B19013_001E': 'tract_median_income'})[
        ['GEOID', 'tract_median_income']
    ]


def assign_tracts(listings: pd.DataFrame) -> pd.Series:
    tracts = gpd.read_file(EXT / 'tl_2019_36_tract' / 'tl_2019_36_tract.shp')
    tracts = tracts[tracts['COUNTYFP'].isin(NYC_COUNTY_FIPS)][['GEOID', 'geometry']]
    tracts = tracts.to_crs('EPSG:4326')

    listings_gdf = gpd.GeoDataFrame(
        listings[['id', 'latitude', 'longitude']],
        geometry=gpd.points_from_xy(listings['longitude'], listings['latitude']),
        crs='EPSG:4326',
    )
    joined = gpd.sjoin(listings_gdf, tracts, how='left', predicate='within')

    # A point on the polygon boundary can match multiple tracts; keep first.
    joined = joined.drop_duplicates(subset='id', keep='first')
    return joined.set_index('id')['GEOID']


def nearest_subway_km(listings: pd.DataFrame) -> np.ndarray:
    sub = pd.read_csv(EXT / 'mta_subway_stations.csv')
    sub = sub.dropna(subset=['gtfs_latitude', 'gtfs_longitude'])
    sub_lat = sub['gtfs_latitude'].values[None, :]
    sub_lon = sub['gtfs_longitude'].values[None, :]
    lst_lat = listings['latitude'].values[:, None]
    lst_lon = listings['longitude'].values[:, None]
    return haversine_km(lst_lat, lst_lon, sub_lat, sub_lon).min(axis=1)


def anchor_distances(listings: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for name, (lat, lon) in ANCHORS_LL.items():
        out[f'dist_{name}_km'] = haversine_km(
            listings['latitude'].values, listings['longitude'].values, lat, lon
        )
    return pd.DataFrame(out, index=listings.index)


def build(verbose: bool = True) -> pd.DataFrame:
    listings = load_listings()
    if verbose:
        print(f'[1/4] loaded {len(listings)} listings')

    tract_geoid = assign_tracts(listings)
    if verbose:
        n = tract_geoid.notna().sum()
        print(f'[2/4] tract assignment: {n}/{len(listings)} ({100*n/len(listings):.2f}%)')

    acs = load_acs_income()
    enriched = listings[['id']].copy()
    enriched['tract_geoid'] = enriched['id'].map(tract_geoid)
    enriched = enriched.merge(acs, left_on='tract_geoid', right_on='GEOID', how='left')
    enriched = enriched.drop(columns='GEOID')
    if verbose:
        n = enriched['tract_median_income'].notna().sum()
        print(f'[3/4] ACS income coverage: {n}/{len(listings)} ({100*n/len(listings):.2f}%)')

    enriched['dist_subway_km'] = nearest_subway_km(listings)
    anchors = anchor_distances(listings)
    enriched = pd.concat([enriched.reset_index(drop=True), anchors.reset_index(drop=True)], axis=1)
    if verbose:
        print(f'[4/4] computed subway + anchor distances')
        print(enriched.describe().round(2).T[['count', 'mean', 'std', 'min', 'max']])

    return enriched


def main():
    DERIVED.mkdir(parents=True, exist_ok=True)
    out_path = DERIVED / 'listings_external.parquet'
    enriched = build(verbose=True)
    enriched.to_parquet(out_path, index=False)
    print(f'\nwrote {out_path} ({out_path.stat().st_size/1024:.1f} KB)')


if __name__ == '__main__':
    main()
