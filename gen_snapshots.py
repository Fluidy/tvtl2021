import pickle
import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from utils import compute_dist, compute_des
matplotlib.rcParams['text.usetex'] = True
matplotlib.rc('font', size=12)

shift_dev = 100  # km
plt_idx = 50  # 631, 580
save = True
random_alt = True

data_dir = 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

num_path = 40
num_planes = 100
num_train = 1000
num_test = 1000
num_time_instance = 100

delta = 0
lb_lat = 25
ub_lat = 55
lb_lon = -40 + delta
ub_lon = -5 + delta
lb_alt = 9
ub_alt = 12

basemap_interval = 5

R = 6371
void_R = 500
void1_lon = (ub_lon - lb_lon)*0.65 + lb_lon  # 65
void1_lat = (ub_lat - lb_lat)*0.5 + lb_lat  # 5
void2_lon = (ub_lon - lb_lon)*0.35 + lb_lon  # 35
void2_lat = (ub_lat - lb_lat)*0.65 + lb_lat  # 65

np.random.seed(1)  # 1


m = Basemap(projection='cyl',
            llcrnrlat=lb_lat,
            urcrnrlat=ub_lat,
            llcrnrlon=lb_lon,
            urcrnrlon=ub_lon,
            resolution='l')

len_lat = R*np.radians(ub_lat - lb_lat)
len_upper_lon = R * np.cos(np.radians(ub_lat)) * np.radians(ub_lon - lb_lon)
len_lower_lon = R * np.cos(np.radians(lb_lat)) * np.radians(ub_lon - lb_lon)
len_vec = np.array([len_lat, len_lat, len_upper_lon, len_lower_lon])
frac_vec = len_vec/np.sum(len_vec)


def get_terminal():
    bound_idx = np.random.uniform()
    if bound_idx < np.sum(frac_vec[:1]):
        lon0 = lb_lon
        lat0 = np.arcsin(np.random.uniform()*(np.sin(np.radians(ub_lat)) - np.sin(np.radians(lb_lat)))
                         + np.sin(np.radians(lb_lat)))*180/np.pi
        idx = 1

    elif bound_idx < np.sum(frac_vec[:2]):
        lon0 = ub_lon
        lat0 = np.arcsin(np.random.uniform() * (np.sin(np.radians(ub_lat)) - np.sin(np.radians(lb_lat)))
                         + np.sin(np.radians(lb_lat)))*180/np.pi
        idx = 2
    elif bound_idx < np.sum(frac_vec[:3]):
        lat0 = ub_lat
        lon0 = np.random.uniform(lb_lon, ub_lon)
        idx = 3
    else:
        lat0 = lb_lat
        lon0 = np.random.uniform(lb_lon, ub_lon)
        idx = 4
    return lat0, lon0, idx


def gen_path(m):
    cnt = 0
    path_list = []
    terminal1_list = []
    terminal2_list = []
    while cnt < num_path:

        lat1, lon1, idx1 = get_terminal()
        lat2, lon2, idx2 = get_terminal()
        while idx2 == idx1:
            lat2, lon2, idx2 = get_terminal()

        line, = m.drawgreatcircle(lon1, lat1, lon2, lat2, c='C0', del_s=10)
        path = line.get_path().vertices
        in_void = False
        for lon, lat in path:
            _, gc_dist, _ = compute_dist(lon, lat, 0, void1_lon, void1_lat, 0)
            if gc_dist < void_R:
                in_void = True
                break
            else:
                _, gc_dist, _ = compute_dist(lon, lat, 0, void2_lon, void2_lat, 0)
                if gc_dist < void_R:
                    in_void = True
                    break
        if not in_void:
            path_list.append(path)
            terminal1_list.append([lat1, lon1])
            terminal2_list.append([lat2, lon2])
            cnt += 1

    return path_list, terminal1_list, terminal2_list


path_list, terminal1_list, terminal2_list = gen_path(m)
with open(data_dir + '/path.pkl', 'wb') as f:
    pickle.dump(path_list, f)


"""
Base map 
"""
fig, ax = plt.subplots(figsize=[5, 4])
# m = Basemap(projection='ortho',
#             lon_0=lb_lon - 20,
#             lat_0=lb_lat,
#             resolution='l')
# m.drawparallels(np.arange(-90, 90, basemap_interval), labels=[1, 0, 0, 0], zorder=1, linewidth=0.25)
# m.drawmeridians(np.arange(-180, 180, basemap_interval), labels=[0, 0, 0, 1], zorder=2, linewidth=0.25)

m.drawcoastlines(linewidth=0.25)
m.fillcontinents(zorder=0)

for i in range(len(path_list)):
    lat1, lon1 = terminal1_list[i]
    lat2, lon2 = terminal2_list[i]
    a = m.drawgreatcircle(lon1, lat1, lon2, lat2, c='C0', linewidth=0.25)


"""
Generate snapshot for training and testing
"""

num_points_vec = np.array([len(i) for i in path_list])
num_planes_vec = np.maximum(np.round(num_planes * num_points_vec/np.sum(num_points_vec)), 1)
node_interval = (num_points_vec/num_planes_vec).astype(int)
max_path = np.argsort(num_planes_vec)[::-1]
diff = num_planes - np.sum(num_planes_vec)
for i in range(np.abs(diff).astype(int)):
    num_planes_vec[max_path[i]] = num_planes_vec[max_path[i]] + np.sign(diff)

start_point = [np.random.choice(len(path)) if num_planes_vec[i] > 0 else None for i, path in enumerate(path_list)]

flight_levels = np.linspace(lb_alt, ub_alt, num_path)


def gen_snapshot(idx, shift=True):
    all_pos = []
    point_path = []
    for i, path in enumerate(path_list):
        if start_point[i] is not None:
            point_path.append(i)
            node = (start_point[i] + idx) % num_points_vec[i]
            if random_alt:
                pos = np.r_[path[node], np.random.uniform(lb_alt, ub_alt)]
            else:
                pos = np.r_[path[node], flight_levels[i]]

            if not shift:
                all_pos.append(pos)
            else:
                d = np.random.rayleigh(shift_dev)
                in_void = True
                while in_void:
                    shifted_pos = compute_des(*pos, np.random.uniform(0, 2 * np.pi), d)
                    _, gc_dist1, _ = compute_dist(shifted_pos[0], shifted_pos[1], 0,
                                                  void1_lon, void1_lat, 0)
                    _, gc_dist2, _ = compute_dist(shifted_pos[0], shifted_pos[1], 0,
                                                  void2_lon, void2_lat, 0)
                    if gc_dist1 > void_R and gc_dist2 > void_R:
                        in_void = False
                all_pos.append(shifted_pos)

            if num_planes_vec[i] > 1:
                for j in range(1, int(num_planes_vec[i])):
                    point_path.append(i)
                    node = (node + node_interval[i]) % num_points_vec[i]
                    if random_alt:
                        pos = np.r_[path[node], np.random.uniform(lb_alt, ub_alt)]
                    else:
                        pos = np.r_[path[node], flight_levels[i]]

                    if not shift:
                        all_pos.append(pos)
                    else:
                        d = np.random.rayleigh(shift_dev)
                        in_void = True
                        while in_void:
                            shifted_pos = compute_des(*pos, np.random.uniform(0, 2 * np.pi), d)
                            _, gc_dist1, _ = compute_dist(shifted_pos[0], shifted_pos[1], 0,
                                                          void1_lon, void1_lat, 0)
                            _, gc_dist2, _ = compute_dist(shifted_pos[0], shifted_pos[1], 0,
                                                          void2_lon, void2_lat, 0)
                            if gc_dist1 > void_R and gc_dist2 > void_R:
                                in_void = False
                        all_pos.append(shifted_pos)

    all_pos = np.array(all_pos)

    # x, y = m(all_pos[:, 0], all_pos[:, 1])
    # ax.scatter(x, y)
    # fig.tight_layout()
    # fig.show()
    return all_pos, point_path


"""
Train Snapshot
"""
snapshots = [gen_snapshot(i % num_time_instance, shift=True)[0] for i in range(num_train)]

if save:
    with open(data_dir + '/train_snapshot.pkl', 'wb') as f:
        pickle.dump(snapshots, f)

x_train, y_train = m(snapshots[plt_idx][:, 0], snapshots[plt_idx][:, 1])
l1 = ax.scatter(x_train, y_train, label=r'Synthetic Flight Position (Train)', c='C1', marker='+', zorder=3)


"""
Test Snapshot
"""
snapshots = [gen_snapshot(i % num_time_instance, shift=True)[0] for i in range(num_test)]

if save:
    with open(data_dir + '/test_snapshot.pkl', 'wb') as f:
        pickle.dump(snapshots, f)
x_test, y_test = m(snapshots[plt_idx][:, 0], snapshots[plt_idx][:, 1])
l2 = ax.scatter(x_test, y_test, label='Synthetic Flight Position (Test)', c='C3', marker='x', zorder=4, s=20)

l3, = ax.plot([0, 0], [0, 0], linewidth=1, label='Preplanned Flight Path', zorder=2)


"""
Planned Position
"""
snapshots = [gen_snapshot(i % num_time_instance, shift=False) for i in range(num_test)]
if save:
    with open(data_dir + '/plot_snapshot.pkl', 'wb') as f:
        pickle.dump(snapshots, f)
x, y = m(snapshots[plt_idx][0][:, 0], snapshots[plt_idx][0][:, 1])
l4 = ax.scatter(x, y, s=1, label='Scheduled Flight Position', zorder=1)

for i in range(len(x_test)):
    # ax.plot([x[i], x_test[i]], [y[i], y_test[i]], linestyle='--', linewidth=0.5,
    #         c='C1', label='Flight Position Deviation' if i == 0 else None)
    region = np.array([compute_des(x[i], y[i], 0, bearing, shift_dev*np.sqrt(-2*np.log(1-0.8)))[:-1]
                       for bearing in np.linspace(0, 2 * np.pi, 20)])
    ax.fill(*m(region[:, 0], region[:, 1]), facecolor='C0', alpha=0.08)


"""
Plot Non-Fly Zones
"""
# void_points = np.array([compute_des(void1_lon, void1_lat, 0, bearing, void_R)[:-1]
#                         for bearing in np.linspace(0, 2*np.pi, 20)])
# x, y = m(void_points[:, 0], void_points[:, 1])
# ax.plot(x, y, c='tab:grey', linestyle='--', linewidth=0.5)
#
# void_points = np.array([compute_des(void2_lon, void2_lat, 0, bearing, void_R)[:-1]
#                         for bearing in np.linspace(0, 2*np.pi, 20)])
# x, y = m(void_points[:, 0], void_points[:, 1])
# ax.plot(x, y, c='tab:grey', linestyle='--', linewidth=0.5)


"""
Plot Figure
"""
meridians = np.arange(lb_lon, ub_lon + basemap_interval, basemap_interval)
meridians_label = ['$' + str(x) + '$' for x in meridians]
parallels = np.arange(lb_lat, ub_lat + basemap_interval, basemap_interval)
parallels_label = ['$' + str(x) + '$' for x in parallels]

ax.set_yticks(parallels)
ax.set_yticklabels(parallels_label)
ax.set_xticks(meridians)
ax.set_xticklabels(meridians_label)
ax.set_xlabel(r'Longitude ($^{\circ}$E)')
ax.set_ylabel(r'Latitude ($^{\circ}$N)')
ax.set_xlim(lb_lon, ub_lon)
ax.set_ylim(lb_lat, ub_lat)
ax.grid()
ax.legend(handles=[l3, l4, l1, l2], loc='center', bbox_to_anchor=(0.55, 0.52),
          fontsize=10).set_zorder(100)
fig.tight_layout()
fig.show()






