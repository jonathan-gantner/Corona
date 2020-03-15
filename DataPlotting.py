# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 22:34:56 2020

@author: pribahsn
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import os

shapefile = os.path.join('data', 'World_map', 'ne_50m_admin_0_countries.shp')


def plotdata(data, title, date=(datetime.today()-timedelta(days=1)).strftime('%#m/%#d/%y'), quantile=0.95):
    # Prepare geopandas shape file for the world and europe
    worldmap = gpd.read_file(shapefile)[['ADMIN', 'ADM0_A3', 'geometry']]
    worldmap.columns = ['country', 'country_code', 'geometry']
    worldmap.drop(worldmap[worldmap['country'] == 'Antarctica'].index, inplace=True)

    europemap = gpd.read_file(shapefile)[['ADMIN', 'ADM0_A3', 'geometry']]
    europemap.columns = ['country', 'country_code', 'geometry']
    europe = pd.read_csv(os.path.join('data', 'World_map', 'europe.csv'))
    europemap = europemap[europemap['country_code'].isin(europe['Country_Code'].to_list())]

    # colormap
    cmap = plt.cm.get_cmap('viridis')
    cmap.set_bad('white', 1.)
    cmap.set_over('red')

    # Map data to map
    dummy = data.loc[date].T.to_frame()
    dummy.reset_index(level=0, inplace=True)
    dummy.columns = ['country_code', 'confirmed']
    worldmap = worldmap.merge(dummy, left_on='country_code', right_on='country_code')
    europemap = europemap.merge(dummy, left_on='country_code', right_on='country_code')

    # # plot the world
    # f, ax = plt.subplots(1)
    # worldmap.plot(ax=ax, linewidth=0.1, edgecolor='0.5', cmap=cmap, legend=True, column='confirmed', vmax=np.nanquantile(europemap.confirmed, q=quantile))
    # ax.set_title(title, fontsize=10)
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_xticks([])
    # ax.set_yticks([])

    # plt.axis('equal')
    # plt.show()

    # plot europe
    f, ax = plt.subplots(1)
    europemap.plot(ax=ax, linewidth=0.1, edgecolor='0.5', cmap=cmap, legend=True, column='confirmed', vmax=np.nanquantile(europemap.confirmed, q=quantile))
    ax.set_title(title, fontsize=10)
    plt.xlim(-25, 45)
    ax.set_ylim(30, 75)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()
