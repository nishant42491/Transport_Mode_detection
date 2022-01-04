import os
import pickle
from math import cos, sin, atan2

import numpy as np
from geopy import distance

class DataEnrich:

    def __init__(self):
        pass

    def _load_raw_pickle(self):
        return pickle.load(open("data/raw_labeled.pkl","rb"))

    def consolidate_trajectories(self):
        raw_dfs = self._load_raw_pickle()
        trajectories = []
        for traj_of_person in raw_dfs:
            dfs_with_label = []
            for traj in traj_of_person:
                if "label" in traj.columns:
                    traj = traj.replace(to_replace='None', value=np.nan).dropna()
                    traj.reset_index(inplace=True)
                    dfs_with_label.append(traj)
            if dfs_with_label:
                trajectories.extend(dfs_with_label)
        return trajectories

    def _calc_speed(self, distance, ts_a, ts_b):
        time_delta = ts_b - ts_a
        if time_delta.total_seconds() == 0:
            return 0
        return distance / time_delta.total_seconds()  # m/s

    def _calc_accel(self, speed_a, speed_b, ts_a, ts_b):
        time_delta = ts_b - ts_a
        speed_delta = speed_b - speed_a
        if time_delta.total_seconds() == 0:
            return 0
        return speed_delta / time_delta.total_seconds()  # m/s^2

    def _calc_jerk(self, acc_a, acc_b, ts_a, ts_b):
        time_delta = ts_b - ts_a
        acc_delta = acc_b - acc_a
        if time_delta.total_seconds() == 0:
            return 0
        return acc_delta / time_delta.total_seconds()

    def _calc_bearing_rate(self, bearing_a, bearing_b, ts_a, ts_b):
        time_delta = ts_b - ts_a
        bear_delta = bearing_b - bearing_a
        if time_delta.total_seconds() == 0:
            return 0
        return bear_delta / time_delta.total_seconds()

    def calc_dist_for_row(self, trajectory_frame, i):
        lat_1 = trajectory_frame["lat"][i-1]
        lat_2 = trajectory_frame["lat"][i]
        if lat_1 > 90:
            print("Faulty", lat_1)
            lat_1 /= 10
        if lat_2 > 90:
            print("Faulty", lat_2)
            lat_2 /= 10

        point_a = (lat_1, trajectory_frame["lon"][i-1])
        point_b = (lat_2, trajectory_frame["lon"][i])
        if point_a[0] == point_b[0] and point_a[1] == point_b[1]:
            trajectory_frame["dist"][i] = 0
        else:
            trajectory_frame["dist"][i] = distance.distance((point_a[0], point_a[1]), (point_b[0], point_b[1])).m

    def calc_speed_for_row(self, trajectory_frame, i):
        trajectory_frame["speed"][i] = self._calc_speed(trajectory_frame["dist"][i],
                                                        trajectory_frame["datetime"][i-1],
                                                        trajectory_frame["datetime"][i]
                                                        )

    def calc_accel_for_row(self, trajectory_frame, i):
        trajectory_frame["accel"][i] = self._calc_accel(trajectory_frame["speed"][i-1],
                                                        trajectory_frame["speed"][i],
                                                        trajectory_frame["datetime"][i - 1],
                                                        trajectory_frame["datetime"][i]
                                                        )

    def set_sample_rate(self, trajectory_frame, min_sec_distance_between_points):
        i = 1
        indices_to_del = []
        deleted = 1
        while i < len(trajectory_frame)-deleted:
            ts1 = trajectory_frame["datetime"][i]
            ts2 = trajectory_frame["datetime"][i+deleted]
            delta = ts2-ts1
            if delta.seconds < min_sec_distance_between_points:
                deleted+=1
                indices_to_del.append(i)
                continue
            i+=deleted
            deleted = 1
        if indices_to_del:
            trajectory_frame.drop(trajectory_frame.index[indices_to_del],inplace=True)
            trajectory_frame.reset_index(inplace=True)

    def set_time_between_points(self, trajectory_frame, i):
        trajectory_frame["timedelta"][i] = (trajectory_frame["datetime"][i]-trajectory_frame["datetime"][i-1]).total_seconds()

    def calc_jerk_for_row(self, trajectory_frame, i):
        trajectory_frame["jerk"][i] = self._calc_jerk(trajectory_frame["accel"][i - 1],
                                                        trajectory_frame["accel"][i],
                                                        trajectory_frame["datetime"][i - 1],
                                                        trajectory_frame["datetime"][i]
                                                        )

    def calc_bearing_for_row(self, trajectory_frame, i):
        a_lat = trajectory_frame["lat"][i - 1]
        a_lon = trajectory_frame["lon"][i - 1]
        b_lat = trajectory_frame["lat"][i]
        b_lon = trajectory_frame["lon"][i]
        x = cos(b_lat) * sin(b_lon-a_lon)
        y = cos(a_lat) * sin(b_lat) - sin(a_lat) * cos(b_lat) * cos(b_lon-a_lon)
        trajectory_frame["bearing"][i] = atan2(x, y)

    def calc_bearing_rate_for_row(self, trajectory_frame, i):
        trajectory_frame["bearing_rate"][i] = self._calc_bearing_rate(trajectory_frame["bearing"][i - 1],
                                                        trajectory_frame["bearing"][i],
                                                        trajectory_frame["datetime"][i - 1],
                                                        trajectory_frame["datetime"][i]
                                                        )

    def calc_features_for_frame(self, traj_frame):
        traj_frame["dist"] = 0
        traj_frame["timedelta"] = 0
        traj_frame["speed"] = 0
        traj_frame["accel"] = 0
        traj_frame["jerk"] = 0
        traj_frame["bearing"] = 0
        traj_frame["bearing_rate"] = 0

        for i, elem in traj_frame.iterrows():
            if i == 0:
                continue
            self.set_time_between_points(traj_frame, i)
            self.calc_dist_for_row(traj_frame, i)
            self.calc_speed_for_row(traj_frame, i)
            self.calc_accel_for_row(traj_frame, i)
            self.calc_jerk_for_row(traj_frame, i)
            self.calc_bearing_for_row(traj_frame, i)
            self.calc_bearing_rate_for_row(traj_frame, i)

    def get_enriched_data(self, from_pickle):
        if from_pickle:
            if os.path.isfile("data/raw_enriched.pkl"):
                print("Reading raw_enriched.pkl")
                return pickle.load(open("data/raw_enriched.pkl", "rb"))
            else:
                print("No pickled enriched dataset, creating. This will take a while.")
        traj = self.consolidate_trajectories()
        for elem in traj:
            self.set_sample_rate(elem, 5)
            self.calc_features_for_frame(elem)
        print("Done, dumping")
        pickle.dump(traj, open("data/raw_enriched.pkl", "wb"))

        return traj


if __name__ == '__main__':
    a=DataEnrich()
    z=a.get_enriched_data(False)
    print(z)
    print("DOneP")



