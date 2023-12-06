import numpy as np
from scipy.sparse import csr_matrix
from utils.dag import DAG

def parse_dot(file_path):
    period_pat = 'T='
    node_pat = '[label=\"'
    edge_pat = '->'

    dag = {}
    wcet = []
    source_node = []
    target_node = []

    with open(file_path, 'r') as f:
        for line in f.readlines():
            if period_pat in line:
                period = int(line.split(period_pat)[1].split('\"')[0])

            elif node_pat in line:
                wcet.append(int(line.split(node_pat)[1].split('\"')[0]))

            elif edge_pat in line:
                lst = line.split(edge_pat)
                source_node.append(int(lst[0]))
                target_node.append(int(lst[1].split(';')[0]))

    wcet = np.array(wcet, dtype=np.int32)
    adj_mat = csr_matrix(
        (np.full_like(source_node, True), 
        (np.array(source_node), np.array(target_node)))).toarray().astype(bool)
    
    dag["wcet"] = wcet
    dag["adj_mat"] = adj_mat
    dag["period"] = period
    return dag


def parse_dot_lf(file_path):
    node_pat = '[label=\"'
    wcet_pat = 'WCET='
    edge_pat = '->'

    dag = {}
    wcet = []
    source_node = []
    target_node = []

    period = 0

    with open(file_path, 'r') as f:
        for line in f.readlines():
            if node_pat in line:
                name = line.split(node_pat)[1].split(',')[0]
                wcet_temp = int(line.split(wcet_pat)[1].split('ms')[0])
                wcet.append(wcet_temp)
                if "Dummy" in name:
                    period += wcet_temp

            elif edge_pat in line:
                lst = line.split(edge_pat)
                source_node.append(int(lst[0]))
                target_node.append(int(lst[1]))

    wcet = np.array(wcet, dtype=np.int32)
    adj_mat = csr_matrix(
        (np.full_like(source_node, True), 
        (np.array(source_node), np.array(target_node)))).toarray().astype(bool)
    
    dag["wcet"] = wcet
    dag["adj_mat"] = adj_mat
    dag["period"] = period
    return dag


def load_dag(sol_path):
    dag = parse_dot(sol_path)
    adj_mat = dag["adj_mat"]
    wcet = dag["wcet"]
    period = dag["period"]
    dag = DAG(adj_mat, wcet, period)
    return dag
