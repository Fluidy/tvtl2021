import numpy as np
from numba import njit

R = 6371


def compute_dist(lon1, lat1, alt1, lon2, lat2, alt2):
    """
    Compute distance and judge connectivity, altitude (km)
    """
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    theta = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    l1 = R + alt1
    l2 = R + alt2

    gc_dist = theta*R
    dist2 = np.maximum(l1 ** 2 + l2 ** 2 - 2 * l1 * l2 * np.cos(theta), 0)
    dist = np.sqrt(dist2)

    sign_cos_alpha1 = l2 ** 2 + dist2 - l1 ** 2
    sign_cos_alpha2 = l1 ** 2 + dist2 - l2 ** 2

    r = l1 * l2 * np.sin(theta) / np.maximum(dist, 1e-8)
    adjacent = (r >= R) | (sign_cos_alpha1 < 0) | (sign_cos_alpha2 < 0) | (np.bitwise_and(dlon == 0, dlat == 0))

    return dist, gc_dist, adjacent


@njit
def floyd(delay_adj):
    delay = np.copy(delay_adj)
    n = len(delay)
    table = -np.ones((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(n):
            if delay[i, j] < np.inf:
                table[i, j] = j

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if delay[i, j] > delay[i, k] + delay[k, j]:
                    delay[i, j] = delay[i, k] + delay[k, j]
                    table[i, j] = table[i, k]
    return delay, table


def compute_des(lon1, lat1, alt1, bearing, d):
    """
    Compute destination point given distance and bearing from start point
    https://www.movable-type.co.uk/scripts/latlong.html
    """
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    delta = (d + alt1)/R
    lat2 = np.arcsin(np.sin(lat1)*np.cos(delta) + np.cos(lat1)*np.sin(delta)*np.cos(bearing))
    lon2 = lon1 + np.arctan2(np.sin(bearing)*np.sin(delta)*np.cos(lat1), np.cos(delta) - np.sin(lat1)*np.sin(lat2))
    return np.array([np.rad2deg(lon2), np.rad2deg(lat2), alt1])
