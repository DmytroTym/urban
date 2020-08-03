"""
This is a module of utilities for manipulation of urban data
See my projects that use this module here:
https://nbviewer.jupyter.org/github/DmytroTym/urban/tree/master/ 
"""

import pandas as pd
import numpy as np
import osmnx as ox
import io
import os
from shapely.geometry import Polygon, Point
from geopy import distance
from collections import defaultdict
from datetime import datetime, timedelta

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
    
    def _streets_from_verts(self, area_verts):
        """Auxilary function"""
        if area_verts is not None:
            area = Polygon(area_verts)
            self.streets = ox.core.osm_net_download(polygon = area,
                                                    network_type = 'all_private')[0]['elements']
            
    def node_coords_from_area(self, node_ids = None):
        """Auxilary function to get latitudes and longitudes of nodes
        In odrer to use it, one needs to pass area_verts first"""
        self.nodecoords = {}
        for row in self.streets:
            if row['type'] == 'node' and (node_ids is None or row['id'] in node_ids):
                self.nodecoords[row['id']] = (row['lat'], row['lon'])
    
    def load_from_file(self, years_months, area_verts = None, node_ids = None, path = ''):
        """
        Function that loads Uber mobility data from specific area.
        
        Arguments:
        years_months - list of tuples (year, month) that are of interest
        area_verts - list of verticles (lon, lat) of area we are interested in
        node_ids - list of OpenStreetMap node ids. if area_verts is None, 
        we use node_ids instead. area_verts and node_ids cannot both be None
        path - path to the folder, where .csv file is located
        """
        
        if area_verts is None and node_ids is None:
            raise ValueError("area_verts and node_ids cannot both be None")
        
        if area_verts is not None:
            node_ids = []
            self._streets_from_verts(area_verts)
            for row in self.streets:
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
        self.data.sort_values(by = ['date', 'hour'], inplace = True)
        del(tp)
        
    def save_data_to_file(self, path):
        """Method to save already filtered data to .csv file"""
        self.data.to_csv(path, index = False)
        
    def load_filtered_from_file(self, path, area_verts = None):
        """Method to load already filtered data from .csv file"""
        tp = []
        self._streets_from_verts(area_verts)
            
        for chunk in pd.read_csv(path, dtype = self._dtypes, parse_dates = ['date'],
                                 chunksize = 100000, low_memory = False):
            tp.append(chunk)

        self.data = pd.concat(tp)
    
    def avg_speed_from_ids(self, id_list):
        """Return average speeds computed over the path given by id_list"""
        adj_node_pairs = [(id_, id_list[i + 1]) for i, id_ in enumerate(id_list[:-1])]
        dists = {pair: distance.distance(self.nodecoords[pair[0]],
                                         self.nodecoords[pair[1]]).km for pair in adj_node_pairs}
        total_dists, total_times = defaultdict(int), defaultdict(int)
        for _, row in self.data.iterrows():
            node_pair = (row['osm_start_node_id'], row['osm_end_node_id'])
            if node_pair in adj_node_pairs:
                total_dists[(row['date'], row['hour'])] += dists[node_pair]
                total_times[(row['date'], row['hour'])] += dists[node_pair]\
                                                           / row['speed_kph_mean']
        
        ave_speeds = {}
        for datetime_key, total_dist in total_dists.items():
            ave_speeds[datetime_key] = total_dist / total_times[datetime_key]
        return ave_speeds


class Visualizer:
    """
    Frequently needed utilities for visualization here
    """
    
    @staticmethod
    def set_style(figsize = (14, 8), style = 'seaborn-darkgrid', fsize = 14):
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
        stds - list of standart deviations, must be same size as data_list
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
    
    _timeFmt = mdates.DateFormatter('%H:%M')
    
    @staticmethod
    def flow_of_traffic_graph(ax, xs, ys, y_ticks, y_names, y_label,
                              title, color = 'blue'):
        """Method for creation of speed versus time and space graph"""
        for x, y in zip(xs, ys):
            ax.plot(x, y, color = color)
        ax.tick_params(axis = 'x', which = 'major', labelsize = 14)
        ax.xaxis.set_major_formatter(Visualizer._timeFmt)
        ax.yaxis.set_ticks(y_ticks)
        ax.yaxis.set_ticklabels(y_names, fontsize = 12)
        ax.set_title(title, fontweight = "bold", size = 16)
        ax.set_ylabel(y_label, fontsize = 12)


class Accident_data:
    """
    TODO: class for storage and manipulation of road accidents data
    """
    
    pass


class GTFS_utils:
    """
    This class contains methods for manipulation of GTFS (general transit feed specification) data
    https://developers.google.com/transit/gtfs/reference/#general_transit_feed_specification_reference
    Class has methods that work with both static and dynamic GTFS data
    """
    
    def __init__(self, static_path = 'gtfs/', dynamic_path = 'Kyiv_realtime_transit_13.07.2020'):
        """
        Initialize class instance by loading data from file
        Dynamic data is supposed to be located in dynamic_path
        """
        self.trips = self._load_and_split(static_path + 'trips.txt')
        self.shapes = self._load_and_split(static_path + 'shapes.txt')
        self.routes = self._load_and_split(static_path + 'routes.txt')
        self.stops = self._load_and_split(static_path + 'stops.txt')
        self.stop_times = self._load_and_split(static_path + 'stop_times.txt')
        self._fill_latlons()
        self._fill_routes_info()
        self.forward_lens, self.forward_pathlens = self._fill_lens(self.forward_latlons)
        self.backward_lens, self.backward_pathlens = self._fill_lens(self.backward_latlons)
        
        self.positions = {}
        directory = os.fsencode(dynamic_path)
        #looping through files with dynamic data and reading them into a dict
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            f = open(os.path.join(dynamic_path, filename), 'r')
            vehicle_list = f.read()[1: -1].split(',')
            for vehicle_position in vehicle_list:
                self._fill_dict(vehicle_position)
            f.close()
        
        self.positions = {k: v for k, v in sorted(self.positions.items(),
                                                  key = lambda x: x[0][2])}

    @staticmethod
    def _load_and_split(filename):
        file_ = io.open(filename, mode = 'r', encoding = 'utf-8')
        return file_.read().split('\n')[1:-1]
    
    @staticmethod
    def _substr_from_str(substr, str_):
        index = str_.find(substr) + len(substr)
        return str_[index:].split('\n', 1)[0]

    def _fill_dict(self, json_str):
        """Since GTFS is not quite JSON-type, we`ll need to manually scrape it"""
        route_id = self._substr_from_str('route_id: ', json_str)[1:-1]
        veh_id = self._substr_from_str('vehicle {\n    id: ', json_str)[1:-1]
        lat = float(self._substr_from_str('latitude: ', json_str))
        lon = float(self._substr_from_str('longitude: ', json_str))
        timestamp = int(self._substr_from_str('timestamp: ', json_str))
        t = datetime.fromtimestamp(timestamp)
        
        self.positions[(route_id, veh_id, t)] = np.array([lat, lon])

    def _fill_latlons(self):
        """Assigning lists of latitude/longitude points to routes"""
        self.forward_latlons = defaultdict(list)
        self.backward_latlons = defaultdict(list)
        self.forward_trip_id, self.backward_trip_id = {}, {}
        
        for route_line in self.routes:
            route_id = route_line.split(',')[0]
            forward_shape_id, backward_shape_id = None, None
            for line in self.trips:
                splited_line = line.split(',')
                if route_id == splited_line[0]:
                    if len(splited_line[3]) > 0 and splited_line[3][-1] == '1':
                        forward_shape_id = splited_line[3]
                        self.forward_trip_id[route_id] = splited_line[2]
                    if len(splited_line[3]) > 0 and splited_line[3][-1] == '0':
                        backward_shape_id = splited_line[3]
                        self.backward_trip_id[route_id] = splited_line[2]

            for line in self.shapes:
                splited_line = line.split(',')
                if forward_shape_id == splited_line[0]:
                    self.forward_latlons[route_id].append((float(splited_line[1]),
                                                           float(splited_line[2])))
                if backward_shape_id == splited_line[0]:
                    self.backward_latlons[route_id].append((float(splited_line[1]),
                                                            float(splited_line[2])))
    
    @staticmethod
    def _fill_lens(latlons):
        """Calculating lengths of routes and sections of routes between consecutive points"""
        lens, pathlens = {}, {}
        for k in latlons:
            lens[k] = [GTFS_utils.dist_km(ll, latlons[k][i]) for i, ll in enumerate(latlons[k][1:])]
            pathlens[k] = sum(lens[k])
        return lens, pathlens
    
    @staticmethod
    def dist_km(p1, p2, simplified = False):
        """simplified flag is True if we want to save time and dont need precise result"""
        if simplified:
            return 80 * np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        return distance.distance(p1, p2).km
    
    @staticmethod
    def angle(p1, p2, p3):
        """angle (in radians) between vectors p1 - p2 and p3 - p2"""
        v1 = p1 - p2
        v2 = p3 - p2

        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.arccos(cosine_angle)

    @staticmethod
    def lenght_of_path(latlons, lens, p, critical_dist = 0.04,
                        critical_angle = 3 * np.pi / 4, simplified = False):
        """This method checks if a point p lies on a path defined by list latlons
        If so, method returns length of path from its beginning to p and index i
        of the last point from latlons_ before p. Otherwise (None, 0) is returned"""
        
        min_dist_to_node, min_ind = None, 0
        for i, ll in enumerate(latlons[:-1]):
            if GTFS_utils.angle(ll, p, latlons[i + 1]) > critical_angle:
                return ((sum(lens[:i]) + GTFS_utils.dist_km(ll, p, simplified)), i)
            dist_to_node = GTFS_utils.dist_km(ll, p, simplified)
            if min_dist_to_node is not None and dist_to_node > min_dist_to_node:
                return sum(lens[:min_ind]), min_ind
            if dist_to_node < critical_dist:
                if min_dist_to_node is None or dist_to_node < min_dist_to_node:
                    min_dist_to_node, min_ind = dist_to_node, i
            
        if GTFS_utils.dist_km(latlons[-1], p, simplified) < critical_dist:
            return (sum(lens), len(latlons) - 1)
        
        return (None, 0)

    def route_fractions_from_id(self, route_id, latlons, lens, pathlen):
        """
        Matching static and dynamic data: assigning vehicle positions to route
        This function returns two dicts, first of which consists of pairs vehicle_id:
        fraction of path completed by this vehicle, second is the same but contains
        times when respective fraction of path was completed
        
        Arguments:
        route_id - id of the route we are interested in
        latlons, lens, pathlen - parameters of path from this route,
        either forward or backward
        """
        ind, times, fractions = defaultdict(int), defaultdict(list), defaultdict(list)

        for k, v in self.positions.items():
            if k[0] == route_id:
                times[k[1]].append(k[2])
                
                len_, i = GTFS_utils.lenght_of_path(latlons[ind[k[1]]:], lens[ind[k[1]]:], v)
                if len_ is not None:
                    fractions[k[1]].append((sum(lens[:ind[k[1]]]) + len_) / pathlen)
                    ind[k[1]] = i
                else:
                    fractions[k[1]].append(None)
                    ind[k[1]] = 0
        return fractions, times
    
    @staticmethod
    def split_fractions_into_trips(fractions, times):
        """
        Quality of data is not always perfect and raw GPS data is not matched to trips
        The objective of this method is to split dicts returned by route_fractions_from_id
        method into separate trips, idealy from one terminal stop to another. This function
        is essentialy a bunch of heuristics and so it looks bad :(
        
        Arguments:
        times, fractions - dicts returned by route_fractions_from_id
        """
        
        firsttime = True
        threshold_dt = timedelta(minutes = 15)
        threshold = 0.05
        
        times_splited, fractions_splited = [], []
        for key in times:
            time_list, frac_list = times[key], fractions[key]
            time_temp, frac_temp = [], []
            for i, item in enumerate(zip(time_list, frac_list)):
                if item[1] is not None and (i == 0 or (item[0] - time_list[i - 1] < threshold_dt and\
                                                       (frac_list[i - 1] is None or item[1] >= frac_list[i - 1]))):
                    frac_temp.append(item[1])
                    time_temp.append(item[0])
                    firsttime = True
                else:
                    if len(time_temp) > 5:
                        if frac_temp[-1] > 1 - threshold and item[1] is None:
                            time_temp.append(item[0])
                            frac_temp.append(1)
                        if frac_temp[0] < threshold and i > len(time_temp):
                            time_temp.insert(0, time_temp[0] - timedelta(minutes = 1))
                            frac_temp.insert(0, 0)
                        if firsttime == False:
                            times_splited.append(time_temp)
                            fractions_splited.append(frac_temp)
                    if firsttime == True:
                        firsttime = False
                        frac_list[i] = frac_list[i - 1]
                        time_list[i] = time_list[i - 1]
                    else:
                        time_temp, frac_temp = [], []
                        
        return fractions_splited, times_splited

    def get_route_stops(self, trip_id, latlons, lens, pathlen):
        """Get stops of a route with given route_id and latlons"""
        stop_names, stop_positions = [], []

        for line in self.stop_times:
            splited_line = line.split(',')
            if trip_id == splited_line[0]:
                stop_id = splited_line[3]
                for line_ in self.stops:
                    splited_line_ = line_.split(',')
                    stop_point = np.array((float(splited_line_[2]), float(splited_line_[3])))
                    if stop_id == splited_line_[0]:
                        len_stop, _ = self.lenght_of_path(latlons, lens, stop_point, critical_dist = 0.3)
                        if len_stop is not None:
                            stop_names.append(splited_line_[1])
                            stop_positions.append(len_stop / pathlen)
                            
        return stop_positions, stop_names
    
    _tr_types = {"Трамвай": 1, "Тролейбус": 2,
                 "Автобус": 3, "Міська електричка": 4}
    
    _no_to_tr_type = {1: "Трамвай", 2: "Тролейбус",
                      3: "Автобус", 4: "Міська електричка"}
    
    def get_route_id(self, transport_type, no):
        """Route id from type of transportation and number"""
        tr_type_no = self._tr_types[transport_type]
        route_id = None
        
        for line in self.routes:
            splited_line = line.split(',')
            if splited_line[1] == no and\
            int(splited_line[3]) == tr_type_no:
                route_id = splited_line[0]
        return route_id
    
    def _fill_routes_info(self):
        """Form a dict with description of each route"""
        self.routes_info = {}
        for line in self.routes:
            splited_line = line.split(',')
            no_ = splited_line[1]
            type_ = self._no_to_tr_type[int(splited_line[3])]
            long_name_ = splited_line[2]
            self.routes_info[splited_line[0]] = (type_, no_, long_name_)

    @staticmethod
    def calc_speed(dist_traveled, t1, t2, timedelta_ = None):
        if timedelta_ is None:
            timedelta_ = (t1 - t2).seconds
        if timedelta_ > 0:
            return 3600 * dist_traveled / timedelta_
        else:
            return None

    def get_cleaned_positions(self, route_id, threshold_dt = timedelta(minutes = 15),
                              stationary_threshold = 0.002, term_thr = 0.1):
        """Clean data from terminal stop waiting, moving to depot, obvious errors etc and return it"""
        prev_pos_time = defaultdict(lambda: None)
        pos_pairs, time_pairs = [], []
        all_dist, all_time = 0, 0
        
        for k, v in self.positions.items():
            if k[0] == route_id:
                prev = prev_pos_time[k[1]]
                for_latlons, back_latlons = self.forward_latlons[k[0]], self.forward_latlons[k[0]]
                for_lens, back_lens = self.forward_lens[k[0]], self.backward_lens[k[0]]
                for_frac, _ = self.lenght_of_path(for_latlons, for_lens, v, simplified = True)
                back_frac, _ = self.lenght_of_path(back_latlons, back_lens, v, simplified = True)
                
                if (for_frac is not None or back_frac is not None) and\
                    self.dist_km(for_latlons[0], v) > term_thr and\
                self.dist_km(for_latlons[-1], v) > term_thr and\
                self.dist_km(back_latlons[0], v) > term_thr and\
                self.dist_km(back_latlons[-1], v) > term_thr and\
                prev is not None and k[2] - prev[1] < threshold_dt:
                    dist_traveled = self.dist_km(v, prev[0])
                    if dist_traveled > stationary_threshold and\
                    self.calc_speed(dist_traveled, k[2], prev[1]) < 100:
                        all_dist += dist_traveled
                        all_time += (k[2] - prev[1]).seconds
                        pos_pairs.append((prev[0], v))
                        time_pairs.append((prev[1], k[2]))
                prev_pos_time[k[1]] = (v, k[2])
                
        return pos_pairs, time_pairs, all_dist, all_time
