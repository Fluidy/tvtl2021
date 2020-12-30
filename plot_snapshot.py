import pickle
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
matplotlib.rcParams['text.usetex'] = True
matplotlib.rc('font', size=12)
from utils import compute_des

delta = 0
lb_lat = 25
ub_lat = 55
lb_lon = -40 + delta
ub_lon = -5 + delta
basemap_interval = 5
g_alt = 5

void_R = 500
void1_lon = (ub_lon - lb_lon)*0.65 + lb_lon  # 65
void1_lat = (ub_lat - lb_lat)*0.5 + lb_lat  # 5
void2_lon = (ub_lon - lb_lon)*0.35 + lb_lon  # 35
void2_lat = (ub_lat - lb_lat)*0.65 + lb_lat  # 65

with open('data/plot_snapshot.pkl', 'rb') as f:
    snapshots = pickle.load(f)
with open('data/path.pkl', 'rb') as f:
    paths = pickle.load(f)


m = Basemap(projection='cyl',
            llcrnrlat=lb_lat,
            urcrnrlat=ub_lat,
            llcrnrlon=lb_lon,
            urcrnrlon=ub_lon,
            resolution='l',
            fix_aspect=False)

fig = plt.figure()
ax = Axes3D(fig, proj_type='ortho')
ax.add_collection3d(m.drawcoastlines(), zs=g_alt - 0.2)
polys = []
for polygon in m.landpolygons:
    polys.append(polygon.get_coords())

# This fills polygons with colors
lc = PolyCollection(polys, edgecolor='black', linewidth=0.3, facecolor='#CCCCCC', alpha=1.0, closed=False)
lcs = ax.add_collection3d(lc, zs=g_alt - 0.2)  # set zero zs#
# # Create underlying blue color rectangle
# # It's `zs` value is -0.003, so it is plotted below land polygons
# bpgon = np.array([[lb_lon, lb_lat],
#        [lb_lon, ub_lat],
#        [ub_lon, ub_lat],
#        [ub_lon, lb_lat]])
# lc2 = PolyCollection([bpgon], edgecolor='none', linewidth=0.1, facecolor='#FFFFFF', alpha=1.0, closed=False)
# lcs2 = ax.add_collection3d(lc2, zs=g_alt-0.003)  # set negative zs value

ax.plot([0, 0], [0, 0], c='tab:grey', linestyle='--', label='No-fly Zone')
ax.plot([0, 0], [0, 0], c='tab:blue', linestyle='-', label='Flight Path')

cnt = 0
for pos, path_idx in zip(*snapshots[0]):
    ax.plot(paths[path_idx][:, 0], paths[path_idx][:, 1], pos[2], c='C0', linewidth=0.25)

    if cnt == 0:
        ax.scatter(pos[0], pos[1], pos[2], c='C0', s=5, label='Airplane')
    else:
        ax.scatter(pos[0], pos[1], pos[2], c='C0', s=5)

    cnt += 1


ax.scatter(-10, 52, g_alt, c='C2', label='Ground Station', marker='^', zorder=0)

meridians = np.arange(lb_lon, ub_lon + basemap_interval, basemap_interval)
parallels = np.arange(lb_lat, ub_lat + basemap_interval, basemap_interval)
ax.set_yticks(parallels)
ax.set_yticklabels(parallels)
ax.set_xticks(meridians)
ax.set_xticklabels(meridians)
ax.set_zlim(g_alt, 13)
ax.set_xlim(lb_lon, ub_lon)
ax.set_ylim(lb_lat, ub_lat)
ax.set_xlabel(r'Longitude ($^{\circ}$E)', labelpad=10)
ax.set_ylabel(r'Latitude ($^{\circ}$N)', labelpad=10)
ax.set_zlabel(r'Altitude (km)')


void_points = np.array([compute_des(void1_lon, void1_lat, 0, bearing, void_R)[:-1]
                        for bearing in np.linspace(0, 2*np.pi, 100)])
x = void_points[:, 0]
y = void_points[:, 1]
z = np.linspace(g_alt, 13, 10)
x_grid, z_grid = np.meshgrid(x, z)
y_grid = np.tile(y, [len(z), 1])
ax.plot(void_points[:, 0], void_points[:, 1], g_alt, c='tab:grey', linestyle='--')
# ax.plot_wireframe(x_grid, y_grid, z_grid, alpha=0.5, color='tab:orange', rcount=5, ccount=0, linestyle='--')


void_points = np.array([compute_des(void2_lon, void2_lat, 0, bearing, void_R)[:-1]
                        for bearing in np.linspace(0, 2*np.pi, 100)])
x = void_points[:, 0]
y = void_points[:, 1]
x_grid, z_grid = np.meshgrid(x, z)
y_grid = np.tile(y, [len(z), 1])
ax.plot(void_points[:, 0], void_points[:, 1], g_alt, c='tab:grey', linestyle='--')
# ax.plot_wireframe(x_grid, y_grid, z_grid, alpha=0.5, color='tab:orange', rcount=5, ccount=0, linestyle='--')

ax.legend(loc='lower left', bbox_to_anchor=(0.12, 0.21)).set_zorder(100)
fig.show()
