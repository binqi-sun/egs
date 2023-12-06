import copy
import numpy as np
import tensorflow as tf
from data_loader.dag_loader import parse_dot_lf as parse_dot
from utils.dag import DAG, padding



class EGSEnv():
    def __init__(self, dag_path, workers=None, out_path=None):
        self.dag_path = dag_path
        self.workers = workers
        self.out_path = out_path
        self.reset()

    def reset(self):
        dag_dict = parse_dot(self.dag_path)
        self._dag = DAG(dag_dict["adj_mat"], dag_dict["wcet"], dag_dict["period"])
        assert self._dag.is_valid(), "EGS: Input DAG is not valid!"
        self.max_n_nodes = self._dag.n_nodes
        self.n_actions = self.max_n_nodes * self.max_n_nodes
        self._init_state()
        self._episode_ended = self.check_episode_end()
        return copy.deepcopy(self._state), self._episode_ended
    
    def get_dag(self):
        return self._dag

    def check_episode_end(self):
        no_action = not np.any(self._state["act_msk"])
        reach_lb = self._dag.width == self._dag.width_lb
        if self.workers is not None:
            reach_lb = self._dag.width <= self.workers
        return True if no_action or reach_lb else False

    def step(self, action: int):
        assert self._state["act_msk"][0, action], "Invalid action!"
        self._update_state(action)
        self._episode_ended = self.check_episode_end()
        return copy.deepcopy(self._state), self._episode_ended

    def _set_state(self):
        wcet_padded = padding(self._dag.node_util, self.max_n_nodes)
        eft_padded = padding(self._dag.eft / self._dag.period, self.max_n_nodes)
        lst_padded = padding(self._dag.lst / self._dag.period, self.max_n_nodes)
        lateral_width = padding(self._dag.lateral_width / self._dag.n_nodes, self.max_n_nodes)
        in_width = padding(self._dag.in_width / self._dag.n_nodes, self.max_n_nodes)
        out_width = padding(self._dag.out_width / self._dag.n_nodes, self.max_n_nodes)

        self._state["nn_input"]["nod_fea"] = tf.cast(tf.expand_dims(
            tf.stack((
                    wcet_padded, 
                    eft_padded, 
                    lst_padded, 
                    lateral_width, 
                    # in_width, 
                    # out_width
                    ), axis=-1), axis=0), dtype=tf.float32)
        self._state["nn_input"]["tra_clo"] = tf.cast(tf.expand_dims(
            padding(self._dag.trans_closure, self.max_n_nodes), axis=0), dtype=tf.bool)
        
        self._state["nn_input"]["act_msk"] = tf.cast(tf.expand_dims(padding(
            self._dag.action_mask, self.max_n_nodes), axis=0), dtype=tf.bool)
        
        self._state["nn_input"]["pad_msk"] = tf.cast(tf.expand_dims(padding(
            np.ones_like(self._dag.wcet, dtype=bool), self.max_n_nodes), axis=0), dtype=tf.bool)

        self._state["act_msk"] = tf.cast(tf.expand_dims(np.reshape(padding(
            self._dag.action_mask, self.max_n_nodes), self.n_actions), axis=0), dtype=tf.bool)
        
    def _init_state(self):
        self._dag.compute_and_set_attributes()
        self._state = {"nn_input": {}}
        self._set_state()

    def _update_state(self, action):
        i_node = action // self.max_n_nodes
        j_node = action % self.max_n_nodes
        self._dag.add_edge_and_update_attributes(i_node, j_node)
        self._set_state()

    def get_result(self):
        return self._dag.width

    def save_dot(self, out_path=None):
        if out_path is None:
            out_path = self.out_path
        node_pat = '[label=\"'
        edge_pat = '->'
        with open(self.dag_path, "r") as f:
            lines = f.readlines()

        virtual_nodes = []
        exec_nodes = []
        for line in lines:
            if node_pat in line:
                i_node = int(line.split(node_pat)[0])
                if "Dummy" in line or "Sync" in line:
                    virtual_nodes.append(i_node)
                else:
                    exec_nodes.append(i_node)

        exec_proc_ass = self._dag.get_proc_assignment(
            self._dag.trans_closure[np.ix_(exec_nodes,exec_nodes)]
        )

        proc_ass = np.zeros(self._dag.n_nodes, dtype=np.int32)
        for ind,ass in enumerate(exec_proc_ass):
            proc_ass[exec_nodes[ind]] = ass + 1

        new_lines = []
        for line in lines:
            if node_pat in line:
                i_node = int(line.split(node_pat)[0])
                node_label = line.split('\"')[1]
                new_node_label = node_label + f", Worker={proc_ass[i_node]}"
                line = line.replace(node_label, new_node_label)
                new_lines.append(line)
            elif edge_pat in line:
                break
            else:
                new_lines.append(line)

        edges = np.transpose(np.nonzero(self._dag.adj_mat))  # [N, 2]
        for edge in edges:
            new_lines.append(f"\t{edge[0]} -> {edge[1]}\n")
        new_lines.append("}")

        with open(out_path, "w") as f:
            f.writelines(new_lines)

        print(f"EGS: output .dot file saved in {out_path}.")

