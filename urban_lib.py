"""
This is a module of utilities for manipulation of some urban data
See my projects that use this module here:
https://nbviewer.jupyter.org/github/DmytroTym/urban/tree/master/ 
"""

import pandas as pd
import numpy as np
import osmnx as ox
from shapely.geometry import Polygon

import matplotlib
import matplotlib.pyplot as plt

class Uber_movement_data:
    """
    A class for storage and filtering of data from:
    https://movement.uber.com/explore/kyiv/speeds/query?lang=en-US
    It's also easy to convert for use in other city, different from Kyiv
    """

    _dtypes = {'year': np.int16, 'month': np.int8, 'day': np.int8, 'hour': np.int8, 
               'osm_start_node_id': np.int64, 'osm_end_node_id': np.int64, 
               'speed_kph_mean': np.float16, 'speed_kph_stddev': np.float16}
    
    @staticmethod
    def datetime_from_row(row):
        """Auxilary function to turn a row of data into pd.datetime"""
        return pd.to_datetime('{}/{}/{}'.format(int(row['year']), int(row['month']), int(row['day'])))
    
    def load_from_file(self, years_months, area_verts = None, node_ids = [], path = ''):
        """
        Function that loads Uber mobility data from specific area.
        
        Arguments:
        years_months - list of tuples (year, month) that are of interest
        area_verts - list of verticles (lon, lat) of area we are interested in
        node_ids - list of OpenStreetMap node ids. if area_verts is None, 
        we use node_ids instead. area_verts and node_ids cannot both be empty
        path - path to the folder, where .csv file is located
        """
        
        if area_verts is None and len(node_ids) == 0:
            raise ValueError("area_verts and node_ids cannot both be empty")
        
        if area_verts is not None:
            area = Polygon(area_verts)
            streets = ox.core.osm_net_download(polygon = area, network_type = 'all_private')[0]['elements']
            for row in streets:
                if row['type'] == 'node':
                    node_ids.append(row['id'])
                    
        filename, tp = 'movement-speeds-hourly-kyiv-', []
        for year, month in years_months:
            full_path = '{}{}{}-{}.csv'.format(path, filename, year, month) 
            for chunk in pd.read_csv(full_path, chunksize = 100000, low_memory = False, 
                                     usecols = self._dtypes.keys(), dtype = self._dtypes):
                filtered_chunk = chunk[(chunk['osm_start_node_id'].isin(node_ids)) &
                                       (chunk['osm_end_node_id'].isin(node_ids))].copy(deep = True)
                #adding new date column for convenience
                if not filtered_chunk.empty:
                    filtered_chunk['date'] = filtered_chunk.apply(lambda row: self.datetime_from_row(row), axis = 1)
                    tp.append(filtered_chunk)

        self.data = pd.concat(tp)
        self.data.sort_values(by = ['date'], inplace = True)
        del(tp)
        
    def save_data_to_file(self, path):
        """Method to save already filtered data to .csv file"""
        self.data.to_csv(path)
        
    def load_filtered_from_file(self, path):
        """Method to load already filtered data from .csv file"""
        tp = []
        for chunk in pd.read_csv(path, dtype = _dtypes, parse_dates = ['date'],
                                 chunksize = 100000, low_memory = False):
            tp.append(chunk)

        self.data = pd.concat(tp)
 
 
class Visualizer:
    """
    Frequently needed utilities for visualization here
    """
    
    @staticmethod
    def set_style(figsize = [14, 8], style = 'seaborn-darkgrid', fsize = 14):
        """Set the style of graphs"""
        matplotlib.rcParams['figure.figsize'] = figsize
        plt.style.use(style)
        plt.rcParams.update({'font.size': fsize})
    
    @staticmethod
    def plot_rolling_averages(data_list, title, colors, xlabel, ylabel, stds = None,
                              window = 0, ylim = None, labels = None, rotation = 0):
        """
        Method to plot two-dimensional data using rolling average
        
        Arguments:
        data_list - list of elements each of which is a tuple (xs, ys)
        where xs is a list (x axis is shared), ys is a list of y-coords
        titles - titles of graphs, must be same size as data_list
        colors - colors of graphs, must be same size as data_list
        window - window of rolling average. If 0, a graph without averaging 
        is built. If type of y is datetime, widow must be timedelta
        """
        if stds is None:
            stds = [0] * len(data_list)
        if labels is None:
            labels = [''] * len(data_list)
        for data, std, label, color in zip(data_list, stds, labels, colors):
            xs, ys = data
            averaged_ys, averaged_stds = [], []
            for x in xs:
                count, sum_, sum_std = 0, 0, 0
                for x_, y_, std_ in zip(xs, ys, std):
                    if x_ >= x - window and x_ <= x + window:
                        count += 1
                        sum_ += y_
                        sum_std += std_
                averaged_ys.append(sum_ / count)
                averaged_stds.append(sum_std / count)
            
            plt.plot(xs, averaged_ys, label = label, color = color)
            plt.fill_between(x = xs, y1 = np.array(averaged_ys) - np.array(averaged_stds),
                             y2 = np.array(averaged_ys) + np.array(averaged_stds),
                             color = color, alpha = 0.15)
        
        if ylim is not None:
            plt.ylim(ylim)
        plt.xticks(rotation = rotation)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if labels[0] != '':
            plt.legend()
        
    @staticmethod
    def put_annotation(coord_x, coord_y, delta_x, caption):
        """Method for annotating graphs"""
        plt.axvspan(coord_x - delta_x, coord_x, facecolor = 'black', alpha = 0.5)
        plt.annotate(caption, va = 'center', xy = (coord_x + delta_x, coord_y),
                     xytext = (coord_x + 20 * delta_x, coord_y),
                     arrowprops = dict(arrowstyle = 'wedge', facecolor = 'black'))
        

class Accident_data:
    """
    TODO: class for storage and manipulation of accidents data
    """
    
    pass