import numpy as np
from mpl_toolkits.basemap import Basemap
from utils import compute_dist, floyd

m = Basemap(projection='cyl')


class AANet(object):
    def __init__(self, snapshot, action_dim, que_delay=5, packet_size=15*8, with_que_delay=True):
        self.with_que_delay = with_que_delay
        self.all_pos = snapshot
        self.all_pos = np.r_[self.all_pos, [[-10, 52, 50/1e3]]]

        self.num_nodes = len(self.all_pos)
        self.action_dim = action_dim
        self.packet_size = packet_size

        lon = self.all_pos[:, 0]
        lat = self.all_pos[:, 1]
        alt = self.all_pos[:, 2]

        self.distance = np.zeros([self.num_nodes, self.num_nodes])
        self.adjacent = np.zeros([self.num_nodes, self.num_nodes], dtype=np.int32)
        for i in range(self.num_nodes):
            self.distance[i], _, self.adjacent[i] = compute_dist(lon, lat, alt, lon[i], lat[i], alt[i])
        np.fill_diagonal(self.adjacent, 1)

        self.link_delay = self.compute_link_delay()
        self.que_delay = que_delay * np.ones(self.num_nodes)
        self.delay = self.link_delay + np.transpose([self.que_delay])
        self.des_node = self.num_nodes - 1
        self.all_candidates, self.all_state, self.all_poss_reward, self.all_poss_done = self.update_observation()

        self.cur_node = None
        self.table = -np.ones_like(self.link_delay, dtype=int)

    def reset(self, src_node):
        self.cur_node = src_node
        return self.cur_node

    def reset_load(self):
        busy_node = np.random.choice(self.num_nodes - 1, 20)
        self.que_delay[busy_node] = 50
        self.delay = self.link_delay + np.transpose([self.que_delay])
        self.update_observation()  # need to update to reward
        return busy_node

    def step(self, action):
        next_node = action
        if self.with_que_delay:
            reward = self.link_delay[self.cur_node, next_node] + self.que_delay[self.cur_node]
        else:
            reward = self.link_delay[self.cur_node, next_node]
        self.cur_node = next_node
        if self.cur_node == self.des_node:
            done = True
        else:
            done = False
        return reward, self.cur_node, done

    def compute_link_delay(self):
        n = len(self.distance)
        link_delay = np.zeros_like(self.distance)
        for i in range(n):
            for j in range(n):
                dist = self.distance[i, j]
                adj = self.adjacent[i, j]
                prop_delay = dist/300  # in millisecond
                if not adj:
                    link_delay[i, j] = np.inf
                else:
                    if dist > 500:
                        trans_delay = self.packet_size/11.016
                    elif dist > 350:
                        trans_delay = self.packet_size/24
                    elif dist > 200:
                        trans_delay = self.packet_size/31.728
                    elif dist > 110:
                        trans_delay = self.packet_size/43.416
                    elif dist > 40:
                        trans_delay = self.packet_size/52.656
                    elif dist > 25:
                        trans_delay = self.packet_size/65.928
                    else:
                        trans_delay = self.packet_size/76.728
                    link_delay[i, j] = prop_delay + trans_delay
        np.fill_diagonal(link_delay, 0)
        return link_delay

    def update_observation(self):
        all_candidates = []
        all_poss_reward = []
        all_poss_done = []
        all_state = np.zeros([self.num_nodes, (self.action_dim + 2)*3])

        for node in range(self.num_nodes):
            if node == self.des_node:
                all_candidates.append([])
            else:
                adj_vec = np.copy(self.adjacent[node])
                adj_vec[node] = 0
                neighbors = np.nonzero(adj_vec)[0]
                neighbors = neighbors[np.argsort(self.distance[neighbors, self.des_node])]
                num_neighbors = len(neighbors)

                if num_neighbors > self.action_dim:
                    all_candidates.append(neighbors[:self.action_dim])
                else:
                    all_candidates.append(neighbors)

            cur_pos = self.all_pos[node]
            des_pos = self.all_pos[self.des_node]
            candidates = all_candidates[node]
            candidates_pos = np.zeros([self.action_dim, 3])
            candidates_pos[:len(candidates)] = self.all_pos[candidates]

            all_state[node] = np.r_[cur_pos, des_pos, candidates_pos.flatten()]
            all_poss_reward.append(self.link_delay[node, candidates] + self.que_delay[candidates])
            poss_done = np.zeros(len(candidates))
            poss_done[candidates == self.des_node] = 1
            all_poss_done.append(poss_done)

        return all_candidates, all_state, all_poss_reward, all_poss_done

    def nearest_neighbor(self):
        nn = []
        for node in range(self.num_nodes):
            adj_vec = self.adjacent[node]
            full_neighbor = np.nonzero(adj_vec)[0]
            nn.append(full_neighbor[np.argmin(self.distance[full_neighbor, self.des_node])])
        return nn

    def floyd_warshall(self):
        if self.with_que_delay:
            self.delay, self.table = floyd(self.delay)
        else:
            self.delay, self.table = floyd(np.copy(self.link_delay))

    def get_opt_path(self, i, j):
        src = i
        des = j
        if self.table[i, j] == -1:
            return None, None
        else:
            path = [i]
            while i != j:
                i = self.table[i, j]
                path.append(i)
            delay = self.delay[src, des]
            return path, delay

    def perimeter_neighbor(self):  # return the neighbors of each node for perimeter forwarding
        p_neighbor = []
        for u in range(self.num_nodes):
            delay_vector = self.link_delay[u]
            full_neighbor = np.nonzero((delay_vector > 0) & (delay_vector < np.inf))[0]  # delay_vector > 0 excludes u
            del_neighbor = []
            for v in full_neighbor:
                for w in full_neighbor:
                    if w is not v:
                        if self.distance[u, v] > np.max([self.distance[u, w], self.distance[v, w]]):
                            del_neighbor.append(v)
                            break
            p_neighbor.append(list(set(full_neighbor) - set(del_neighbor)))
        return p_neighbor

    def get_nodes_within_radius(self, r):
        dist, _, _ = compute_dist(self.all_pos[:, 0], self.all_pos[:, 1], self.all_pos[:, 2],
                                  -25, 25, 10)
        return np.where(dist < r)[0]


