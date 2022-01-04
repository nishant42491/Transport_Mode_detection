import os
import pickle
import pandas as pd
import datetime
import sys

from multiprocessing import Pool

def get_labeled_data_as_df(path):
    trajectory_frames = []

    labelfile = os.path.join(path, "labels.txt")
    _label_df = pd.read_csv(labelfile,sep="\t",header=0,names=["starttime", "endtime", "mode"],parse_dates=[0,1])
    _label_df["startdate"] = _label_df["starttime"].dt.date
    _label_startdate_set = set(_label_df["startdate"])

    datapath = os.path.join(path, "Trajectory")
    for file in os.listdir(datapath):
        df = pd.read_csv(os.path.join(datapath,file),
                         sep=",",
                         header=None,
                         skiprows=6,
                         usecols=[0, 1, 3, 5, 6],
                         names=["lat", "lon", "altitude", "date", "time"])

        df["datetime"] = pd.to_datetime(df['date'] + ' ' + df['time'])
        date_of_traj = datetime.datetime.strptime(file[:8],"%Y%m%d").date()

        if date_of_traj in _label_startdate_set:
            labels_for_date = _label_df[_label_df["startdate"] == date_of_traj]

            def is_in(trajrow):
                for i, row in labels_for_date.iterrows():
                    if row["starttime"] <= trajrow["datetime"] <= row["endtime"]:
                        return row["mode"]

            df["label"] = df.apply(is_in, axis=1)

        trajectory_frames.append(df)
        print("added", datapath, file)
    return trajectory_frames

if __name__ == '__main__':
    '''if len(sys.argv) < 2:
        print("Usage: raw_data_loader.py /path/to/geolife/Data/")
        exit(-1)'''
    path = 'D:\Geolife Trajectories 1.3\Geolife Trajectories 1.3\Data'
    traj_with_labels_paths = []
    for file in os.listdir(path):
        currfile = os.path.join(path, file)
        if os.path.isdir(currfile):
            if "labels.txt" not in os.listdir(currfile):
                continue
            traj_with_labels_paths.append(currfile)

    with Pool(3) as p:
        traj_frames = p.map(get_labeled_data_as_df, traj_with_labels_paths)

    pickle.dump(traj_frames, open( "data/raw_labeled.pkl", "wb"))