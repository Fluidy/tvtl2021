import numpy as np
from utils import compute_dist


class Packet(object):
    def __init__(self, D):
        self.D = D
        self.h = None
        self.Lp = None
        self.Lf = None
        self.e0 = None
        self.M = "greedy"


def norm(x):
    if x < 0:
        x = x + 2*np.pi
    return x


def intersect(a, b, c, d, locations):
    x = None
    y = None
    if (max(locations[a, 0], locations[b, 0]) >= min(locations[c, 0], locations[d, 0])
            and max(locations[c, 0], locations[d, 0]) >= min(locations[a, 0], locations[b, 0])
            and max(locations[a, 1], locations[b, 1]) >= min(locations[c, 1], locations[d, 1])
            and max(locations[c, 1], locations[d, 1]) >= min(locations[a, 1], locations[b, 1])):  # 排除共线不相交

        ac = [locations[c, 0] - locations[a, 0], locations[c, 1] - locations[a, 1]]
        ad = [locations[d, 0] - locations[a, 0], locations[d, 1] - locations[a, 1]]
        bc = [locations[c, 0] - locations[b, 0], locations[c, 1] - locations[b, 1]]
        bd = [locations[d, 0] - locations[b, 0], locations[d, 1] - locations[b, 1]]
        if (np.cross(ac, ad)*np.cross(bc, bd) <= 0) and (np.cross(ac, bc)*np.cross(ad, bd) <= 0):  # 共线不相交会返回1
            A1 = locations[b, 1] - locations[a, 1]
            B1 = locations[a, 0] - locations[b, 0]
            C1 = locations[b, 0]*locations[a, 1] - locations[a, 0]*locations[b, 1]
            A2 = locations[d, 1] - locations[c, 1]
            B2 = locations[c, 0] - locations[d, 0]
            C2 = locations[d, 0]*locations[c, 1] - locations[c, 0]*locations[d, 1]
            x = (C2*B1 - C1*B2)/(A1*B2 - A2*B1)
            y = (C1*A2 - C2*A1)/(A1*B2 - A2*B1)

    return [x, y]


def distance(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def face_change(packet, cur_node, t, p_neighbors, locations):
    i = intersect(t, cur_node, packet.Lp, packet.D, locations)
    if i != [None, None]:
        if distance(i, locations[packet.D]) <= distance(packet.Lf, locations[packet.D]) + 1e-10:
            packet.Lf = i
            t = right_hand_forward(cur_node, t, p_neighbors, locations)
            try:
                t = face_change(packet, cur_node, t, p_neighbors, locations)
            except RecursionError:
                t = t
            packet.e0 = (cur_node, t)
    return t


def right_hand_forward(cur_node, baseline_start, p_neighbors, locations):
    b_in = norm(np.arctan2(locations[cur_node, 1] - locations[baseline_start, 1],
                           locations[cur_node, 0] - locations[baseline_start, 0]))
    delta_min = 3*np.pi
    a_min = cur_node
    for a in p_neighbors[cur_node]:
        if len(p_neighbors[cur_node]) != 1:
            if a != baseline_start:
                b_a = norm(np.arctan2(locations[cur_node, 1] - locations[a, 1],
                                      locations[cur_node, 0] - locations[a, 0]))
                delta_b = norm(b_a - b_in)
                if delta_b < delta_min:
                    delta_min = delta_b
                    a_min = a
        else:
            a_min = p_neighbors[cur_node][0]
    return a_min


def peri_init_forward(cur_node, baseline_start, p_neighbors, locations):
    b_in = norm(np.arctan2(locations[cur_node, 1] - locations[baseline_start, 1],
                           locations[cur_node, 0] - locations[baseline_start, 0]))
    delta_min = 3*np.pi
    a_min = cur_node
    for a in p_neighbors[cur_node]:
        b_a = norm(np.arctan2(locations[cur_node, 1] - locations[a, 1],
                              locations[cur_node, 0] - locations[a, 0]))
        delta_b = norm(b_a - b_in)
        if delta_b < delta_min and a != cur_node:  # IMPORTANT: a != cur_node
            delta_min = delta_b
            a_min = a
    return a_min


def gpsr_forward(packet, cur_node, nearest_neighbor, p_neighbors, locations, all_pos3d):
    t = nearest_neighbor[cur_node]
    if packet.M == "greedy":
        if t == cur_node:
            packet.M = "perimeter"
            packet.Lp = cur_node
            packet.Lf = locations[cur_node]
            t = peri_init_forward(cur_node, packet.D, p_neighbors, locations)
            packet.e0 = (cur_node, t)
        packet.h = cur_node
        return t
    else:
        # if distance(locations[cur_node], locations[packet.D]) < distance(locations[packet.Lp], locations[packet.D]) \
        #         and t != cur_node:

        dist_cur_node, _, _ = compute_dist(*all_pos3d[cur_node], *all_pos3d[packet.D])
        dist_Lp, _, _ = compute_dist(*all_pos3d[packet.Lp], *all_pos3d[packet.D])
        if dist_cur_node < dist_Lp and t != cur_node:
            packet.h = cur_node
            return t
        else:
            t = right_hand_forward(cur_node, packet.h, p_neighbors, locations)
            if packet.e0 == (cur_node, t):
                packet.h = cur_node
                return cur_node  # None unreachable
            else:
                t = face_change(packet, cur_node, t, p_neighbors, locations)
                packet.h = cur_node
                return t
