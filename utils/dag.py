import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


def padding(array, length_after_pad):
    dim = len(array.shape)
    current_length = array.shape[0]
    pad_width = length_after_pad - current_length
    if pad_width > 0:
        if dim == 1:
            padded_array = np.pad(array, (0, pad_width))
        elif dim == 2:
            padded_array = np.pad(array, ((0, pad_width), (0, pad_width)))
    else:
        padded_array = array
    return padded_array

class DAG():
    def __init__(self, adj_mat, wcet, period):
        self.adj_mat = adj_mat
        self.wcet = wcet
        self.period = period
        assert self.adj_mat.shape[0] == self.adj_mat.shape[1] == self.wcet.shape[0]

        self.n_nodes = len(wcet)
        self.node_util = wcet / period
        self.graph_util = np.sum(self.wcet) / self.period
        self.trans_closure = self.transitive_closure()
        self.adj_mat = self.transitive_reduction()

        self.eft = self.wcet
        self.lst = self.period - self.wcet

    def compute_and_set_attributes(self, include_implicit_edge=False):
        self.eft = self.compute_earliest_finishing_time()
        self.est = self.eft - self.wcet
        self.lst = self.compute_latest_starting_time()
        self.lft = self.lst + self.wcet
        
        if include_implicit_edge:
            self.adj_mat |= self.get_implicit_edge_mask() # make implicit edge explicit
            self.trans_closure = self.transitive_closure()
            self.adj_mat = self.transitive_reduction()
        
        self.length = self.compute_dag_length()
        self.density = self.compute_dag_density()
        self.width = self.compute_dag_width()
        self.in_degree = self.get_in_degree()
        self.out_degree = self.get_out_degree()
        self.lateral_width, self.in_width, self.out_width = self.compute_node_dop()
        self.critical_nodes, self.critical_edges = self.get_critical_nodes_edges()
        self.width_lb = self.get_width_lower_bound()
        self.edge_exist_mask = self.get_edge_exist_mask()
        self.cycle_mask = self.get_cycle_mask()
        self.time_inversion_mask = self.get_time_inversion_mask()
        self.action_mask_without_critical_edges = self.get_action_mask_without_critical_edges()
        self.action_mask = self.get_action_mask()

    def add_edge_and_update_attributes(self, i_node, j_node, include_implicit_edge=False):
        self.adj_mat[i_node, j_node] = True
        if include_implicit_edge:
            self.adj_mat |= self.get_implicit_edge_mask() # make implicit edge explicit
            self.trans_closure = self.transitive_closure()
        else:
            self.update_transitive_closure(i_node, j_node)
        self.adj_mat = self.transitive_reduction()
        self.compute_and_set_attributes()

    def transitive_closure(self):
        identity_like_adj_mat = np.identity(self.n_nodes, dtype=bool)
        closure = identity_like_adj_mat | self.adj_mat
        # Floyd-Warshall Algorithm
        for k in range(self.n_nodes):
            for i in range(self.n_nodes):
                for j in range(self.n_nodes):
                    closure[i, j] = closure[i, j] or (closure[i, k] and closure[k, j])
        closure = ~identity_like_adj_mat & closure
        return closure

    def transitive_reduction(self):
        return self.adj_mat & (~(np.matmul(self.trans_closure, self.adj_mat)))

    def update_transitive_closure(self, i_node, j_node):
        self.trans_closure[i_node, j_node] = True
        self.trans_closure[i_node, :] |= self.trans_closure[j_node, :]
        for k in range(self.n_nodes):
            if self.trans_closure[k, i_node]:
                self.trans_closure[k, :] |= self.trans_closure[j_node, :]
                self.trans_closure[k, j_node] = True

    def forward_propagate(self, est):
        eft = est + self.wcet
        est_next = np.empty(self.n_nodes, dtype=np.int32)
        for i in range(self.n_nodes):
            est_next[i] = np.max(self.adj_mat[:, i] * eft)
        return est_next

    def backward_propagate(self, lft):
        lst = lft - self.wcet
        lft_next = np.empty(self.n_nodes, dtype=np.int32)
        for i in range(self.n_nodes):
            lft_next[i] = np.min(np.where(self.adj_mat[i, :], lst, self.period))
        return lft_next

    def compute_earliest_finishing_time(self):
        est = self.eft - self.wcet
        for _ in range(self.n_nodes):
            est_last = est
            est = self.forward_propagate(est)
            if (np.array_equal(est_last, est)):
                break
        eft = est + self.wcet
        return eft

    def compute_latest_starting_time(self):
        lft = self.lst + self.wcet
        for _ in range(self.n_nodes):
            lft_last = lft
            lft = self.backward_propagate(lft)
            if (np.array_equal(lft_last, lft)):
                break
        lst = lft - self.wcet
        return lst

    def compute_dag_length(self):
        return np.max(self.eft)

    def compute_dag_density(self):
        return self.length / self.period

    def compute_dag_width(self, trans_closure=None):
        if trans_closure is None:
            trans_closure = self.trans_closure
        dag = csr_matrix(trans_closure, dtype=bool)
        perm = maximum_bipartite_matching(dag, perm_type='row')
        # count the # of `-1`
        num_unmatched_edges = list(perm).count(-1)
        # # paths + # matched edges = n, so # paths == # unmatched edges
        return num_unmatched_edges

    def compute_node_dop(self):
        trans_closure_tiled = np.tile(np.expand_dims(self.trans_closure, axis=0), [self.n_nodes, 1, 1])
        ancestors_mask = trans_closure_tiled[np.arange(self.n_nodes), :, np.arange(self.n_nodes)] | np.identity(self.n_nodes, dtype=bool)
        descendants_mask = trans_closure_tiled[np.arange(self.n_nodes), np.arange(self.n_nodes), :] | np.identity(self.n_nodes, dtype=bool)
        reachable_node_mask = ancestors_mask | descendants_mask
        return self._compute_node_dop(reachable_node_mask, trans_closure_tiled), self._compute_node_dop(descendants_mask, trans_closure_tiled), self._compute_node_dop(ancestors_mask, trans_closure_tiled)

    def _compute_node_dop(self, isolate_node_mask, trans_closure_tiled):
        n_isolate_nodes = np.sum(isolate_node_mask, axis=-1)

        forward_reachable_mask = np.tile(np.expand_dims(isolate_node_mask, axis=-1), [1,1,self.n_nodes])
        backward_reachable_mask = np.transpose(forward_reachable_mask, [0,2,1])
        isolate_mask = ~forward_reachable_mask & ~backward_reachable_mask
        trans_closure_tiled = trans_closure_tiled & isolate_mask

        node_dop = np.zeros(self.n_nodes, dtype=np.int32)
        for i in range(self.n_nodes):
            node_dop[i] = self.compute_dag_width(trans_closure_tiled[i]) - n_isolate_nodes[i]
        return node_dop

    def check_schedulability(self, n_processors):
        width = self.compute_dag_width()
        length = self.compute_dag_length()
        return (width <= n_processors) and (length <= self.period)

    def is_dag(self):
        cycle_mask = self.get_cycle_mask()
        return not np.any(~cycle_mask & self.trans_closure)

    def is_valid(self):
        self.eft = self.compute_earliest_finishing_time()
        return self.is_dag() and self.compute_dag_length() <= self.period

    def get_in_degree(self):
        # return np.sum(self.adj_mat, axis=0)
        return np.sum(self.trans_closure, axis=0)

    def get_out_degree(self):
        # return np.sum(self.adj_mat, axis=-1)
        return np.sum(self.trans_closure, axis=-1)

    def get_width_lower_bound(self):
        critical_est = np.min(self.est[self.critical_nodes])
        critical_lft = np.max(self.lft[self.critical_nodes])
        critical_period = critical_lft - critical_est
        critical_lb = np.int32(np.ceil(np.sum(self.wcet[self.critical_nodes]) / critical_period))
        return np.maximum(critical_lb, np.int32(np.ceil(self.graph_util)))

    def get_time_inversion_mask(self):
        return np.expand_dims(self.eft, axis=-1) <= np.expand_dims(self.lst, axis=0)

    def get_implicit_edge_mask(self):
        return np.expand_dims(self.lft, axis=-1) <= np.expand_dims(self.est, axis=0)

    def get_cycle_mask(self):
        return (~np.transpose(self.trans_closure)) & (~np.identity(self.n_nodes, dtype=bool))

    def get_edge_exist_mask(self):
        return ~self.trans_closure

    def get_critical_nodes_edges(self):
        critical_nodes = self.lateral_width == (self.width - 1)
        critical_edges = np.expand_dims(critical_nodes, axis=-1) & np.expand_dims(critical_nodes, axis=0) & ~np.identity(self.n_nodes, dtype=bool)
        return critical_nodes, critical_edges

    def get_action_mask_without_critical_edges(self):
        return self.time_inversion_mask & self.edge_exist_mask & self.cycle_mask

    def get_action_mask(self):
        return self.time_inversion_mask & self.edge_exist_mask & self.cycle_mask & self.critical_edges

    def get_min_path_cover(self, trans_closure=None):
        if trans_closure is None:
            trans_closure = self.trans_closure
        dag = csr_matrix(trans_closure, dtype=bool)
        perm = maximum_bipartite_matching(dag, perm_type='row')
        perm = list(perm)
        paths = []
        for ind,val in enumerate(perm):
            if val == -1:
                path = [ind]
                while ind in perm:
                    ind = perm.index(ind)
                    path.append(ind)
                paths.append(path)
        return paths
    
    def get_proc_assignment(self, trans_closure=None):
        if trans_closure is None:
            trans_closure = self.trans_closure
        paths = self.get_min_path_cover(trans_closure=trans_closure)
        proc_ass = np.zeros(shape=(len(paths), self.n_nodes), dtype=bool)
        for i,path in enumerate(paths):
            proc_ass[i][path] = True
        proc_ass = proc_ass.T
        proc_ass = np.nonzero(proc_ass)[1]
        return proc_ass


