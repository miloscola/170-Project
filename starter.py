import networkx as nx
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from pathlib import Path
from tqdm.auto import tqdm
import tarfile
import random

# Scoring constants
MAX_WEIGHT = 1000
MAX_EDGES = 10000
N_SMALL = 100
N_MEDIUM = 300
N_LARGE = 1000
K_EXP = 0.5
K_COEFFICIENT = 100
B_EXP = 70

INPUT_SIZE_LIMIT = 1000000
OUTPUT_SIZE_LIMIT = 10000


def write_input(G: nx.Graph, path: str, overwrite: bool=False):
    assert overwrite or not os.path.exists(path), \
        'File already exists and overwrite set to False. Move file or set overwrite to True to proceed.'
    if validate_input(G):
        with open(path, 'w') as fp:
            json.dump(nx.node_link_data(G), fp)


def read_input(path: str):
    assert os.path.getsize(path) < INPUT_SIZE_LIMIT, 'This input file is too large'
    with open(path) as fp:
        G = nx.node_link_graph(json.load(fp))
        if validate_input(G):
            return G


def write_output(G: nx.Graph, path: str, overwrite=False):
    assert overwrite or not os.path.exists(path), \
        'File already exists and overwrite set to False. Move file or set overwrite to True to proceed.'
    if validate_output(G):
        with open(path, 'w') as fp:
            json.dump([G.nodes[v]['team'] for v in range(G.number_of_nodes())], fp)


def read_output(G: nx.Graph, path: str):
    assert os.path.getsize(path) < OUTPUT_SIZE_LIMIT, 'This output file is too large'
    with open(path) as fp:
        l = json.load(fp)
        assert isinstance(l, list), 'Output partition must be a list'
        assert set(G) == set(range(len(l))), 'Output does not match input graph'
        nx.set_node_attributes(G, {v: l[v] for v in G}, 'team')
        if validate_output(G):
            return G


def validate_graph(G: nx.Graph):
    assert not G.is_directed(), 'G should not be directed'
    assert set(G) == set(range(G.number_of_nodes())), 'Nodes must be numbered from 0 to n-1'
    return True


def validate_input(G: nx.Graph):
    for n, d in G.nodes(data=True):
        assert not d, 'Nodes cannot have data'
    for u, v, d in G.edges(data=True):
        assert u != v, 'Edges should be between distinct vertices (a penguin is experiencing inner-conflict)'
        assert set(d) == {'weight'}, 'Edge must only have weight data'
        assert isinstance(d['weight'], int), 'Edge weights must be integers'
        assert d['weight'] > 0, 'Edge weights must be positive'
        assert d['weight'] <= MAX_WEIGHT, f'Edge weights cannot be greater than {MAX_WEIGHT}'
    assert G.number_of_edges() <= MAX_EDGES, 'Graph has too many edges'
    assert sum(d for u, w, d in G.edges(data='weight')) >= MAX_WEIGHT*MAX_EDGES*0.05, \
        f'There must be at least {MAX_WEIGHT*MAX_EDGES*0.05} edge weight in the input.'
    return validate_graph(G)


def validate_output(G: nx.Graph):
    for n, d in G.nodes(data=True):
        assert set(d) == {'team'}, 'Nodes must have team data'
        assert isinstance(d['team'], int), 'Team identifier must be an integer'
        assert d['team'] > 0, 'Team identifier must be greater than 0'
        assert d['team'] <= G.number_of_nodes(), 'Team identifier unreasonably large'
    return validate_graph(G)


def score(G: nx.Graph, separated=False):
    output = [G.nodes[v]['team'] for v in range(G.number_of_nodes())]
    teams, counts = np.unique(output, return_counts=True)

    k = np.max(teams)
    b = np.linalg.norm((counts / G.number_of_nodes()) - 1 / k, 2)
    C_w = sum(d for u, v, d in G.edges(data='weight') if output[u] == output[v])

    if separated:
        return C_w, K_COEFFICIENT * math.exp(K_EXP * k), math.exp(B_EXP * b)
    return C_w + K_COEFFICIENT * math.exp(K_EXP * k) + math.exp(B_EXP * b)


def visualize(G: nx.Graph):
    output = G.nodes(data='team', default=0)
    partition = dict()
    for n, t in output:
        if t not in partition:
            partition[t] = []
        partition[t].append(n)

    pos = dict()
    circle_size = len(partition) * 0.5
    for k, v in partition.items():
        pos.update(nx.shell_layout(G, nlist=[v], center=(circle_size*math.cos(math.tau*k / len(partition)),
                                                         circle_size*math.sin(math.tau*k / len(partition)))))

    crossing_edges = [e for e in G.edges(data='weight') if output[e[0]] != output[e[1]]]
    within_edges = [e for e in G.edges(data='weight') if output[e[0]] == output[e[1]]]
    max_weight = max(nx.get_edge_attributes(G, name='weight').values())

    nx.draw_networkx_nodes(G, pos, node_color=[output[n] for n in G],
                           cmap=cm.get_cmap('tab20b'))
    nx.draw_networkx_labels(G, pos, font_size=10, font_color="white")

    nx.draw_networkx_edges(G, pos, edgelist=crossing_edges, edge_color=[x[2] for x in crossing_edges],
                           edge_cmap=cm.get_cmap('Blues'), edge_vmax=max_weight*1.5, edge_vmin=max_weight*-0.2)
    nx.draw_networkx_edges(G, pos, width=2, edgelist=within_edges, edge_color=[x[2] for x in within_edges],
                           edge_cmap=cm.get_cmap('Reds'), edge_vmax=max_weight*1.5, edge_vmin=max_weight*-0.2)

    plt.tight_layout()
    plt.axis("off")
    plt.show()


def run(solver, in_file: str, out_file: str, overwrite: bool=False):
    instance = read_input(in_file)
    output = solver(instance)
    if output:
        instance = output
    write_output(instance, out_file, overwrite)
    print(f"{str(in_file)}: cost", score(instance))


def run_all(solver, in_dir, out_dir, overwrite: bool=False):
    for file in tqdm([x for x in os.listdir(in_dir) if x.endswith('.in')]):
        run(solver, str(Path(in_dir) / file), str(Path(out_dir) / f"{file[:-len('.in')]}.out"), overwrite)


def tar(out_dir, overwrite=False):
    path = f'{os.path.basename(out_dir)}.tar'
    assert overwrite or not os.path.exists(path), \
        'File already exists and overwrite set to False. Move file or set overwrite to True to proceed.'
    with tarfile.open(path, 'w') as fp:
        fp.add(out_dir)
        
def make_rand_graph(nodes, edges):
    g = nx.gnm_random_graph(nodes, edges)
    for (u, v) in g.edges:
        g.edges[u,v]['weight'] = random.randint(1, 1000)
    return g   

def generate_inputs():
    one_hundred_node = make_rand_graph(100, 1000)
    validate_input(one_hundred_node)
    three_hundred_node = make_rand_graph(300, 1000)
    validate_input(three_hundred_node)
    one_thousand_node = make_rand_graph(1000, 1000)
    validate_input(one_thousand_node)

    write_input(one_hundred_node, 'C:/Users/Milo/Documents/cs170/project/inputs/100_node.in', overwrite=True)
    write_input(three_hundred_node, 'C:/Users/Milo/Documents/cs170/project/inputs/300_node.in', overwrite=True)
    write_input(one_thousand_node, 'C:/Users/Milo/Documents/cs170/project/inputs/1000_node.in', overwrite=True)

#Simulated annealing alg: start off with penguins randomly devided into k teams evenly. Use
#simulated annealing to optomize arangment. Run simulated anealing multiple times to look
#at different graph sizes. Terminate poorly performing team numbers early


def update(G: nx.Graph, i, j, v, p, cw):
    #p is a list of team sizes
    #i is team i
    #j is team j
    #v is vertex moving from i to j
    #G is graph
    #cw is given

    k = len(p)
    b = np.linalg.norm((p / G.number_of_nodes()) - 1 / k, 2)
    
    biold = p[i] / G.number_of_nodes() - (1 / k)
    bjold= p[j] / G.number_of_nodes() - (1 / k )
    
    binew = biold - (1/G.number_of_nodes())
    bjnew = bjold - (1/G.number_of_nodes())
    cp = np.exp(70*math.sqrt((b**2)-(biold**2)-(bjold**2)-(binew**2)-(bjnew**2)))

    v_adj = G.neighbors(v)
    for neighbor in v_adj:
        if G.nodes[neighbor]['team'] == i:
            cw -= G.edges[v,neighbor]['weight']
        elif G.nodes[neighbor]['team'] == i:
            cw += G.edges[v,neighbor]['weight']

    return cw+(100*math.exp(0.5*k))+cp


#k = number of teams
#pi = partition of penguins to teams (dict mapping vertex to team)
#p = vector of team sizes (array with position 0 = size of team 1 ... etc)
#C = cost of partition 
#G = graph
#t_start = start time
#t_end = end time
#T_func = temprature function
def simulated_annealing(k, p, G, t_start, t_end, T_func):
    cur = G.copy()
    C = score(cur)
    for t in range(t_start, t_end):
        #update temp
        T = T_func(t)
        if T <= 0:
            return cur
        #randomly select new move
        v = random.randint(0, G.size() - 1)
        old_team = cur[v]['team']
        new_team = random.randint(1, k)
        #ensure new team is diff from old team
        while new_team == old_team:
            new_team = random.randint(1, k)
        p[old_team - 1] -= 1
        p[new_team - 1] += 1
        cur[v]['team'] = new_team
        new_C = score(cur)
        delta_C = new_C - C
        if delta_C < 0: 
            C = new_C
        else:
            if random.random() < math.exp(math.e, -1*delta_C/T):
                C = new_C
            else:
                #change parameters back
                cur[v]['team'] = old_team
                p[old_team - 1] += 1
                p[new_team - 1] -= 1
        return (cur, C, p)

#partitions: a list of arrays [number of partitions, p, graph, score]
#time_iter: an iterator that returns the ammount of time spent on each recursive step
#start_time: start time for temp iterator
def recurse(partitions, depth, t_start, t_func, T_func):
    #run simulated annealing on each partition
    t_end = t_start + t_func(depth)
    for par in partitions:
        SI = simulated_annealing(par[0], par[1], par[2], t_start, t_end, T_func)
    #if this is the last partition, return it
    if len(partitions) == 1:
        return partitions[0][2]
    #remove the lower scoring half of the partitions
    partitions.sort(key=lambda x: x[3])
    half = partitions[len(partitions)/2:]
    #recursively call solve until there are no 
    return recurse(half, depth + 1, t_end, t_func, T_func)

def pre_set(G):
    for v in G:
        v['team'] = 1

#TESTS
def generate_test_graphs(nodes, num, dest):   
    for i in range(num):
        graph = make_rand_graph(nodes, 1000)
        with open(dest + '/' + str(i) + '.in', 'w') as fp:
            json.dump(nx.node_link_data(graph), fp)
    
#generate_test_graphs(100, 3, 'C:/Users/Milo/Documents/cs170/project/Test_Graphs/Large')        

def sanity_check():
    G = read_input('C:/Users/Milo/Documents/cs170/project/Test_Graphs/Large/0.in')
    print('oringinial score is:')
    print(score(G))
    G1 = G.copy()
    pre_set(G1)
    G2 = G.copy()
    pre_set(G2)
    G3 = G.copy()
    pre_set(G3)
    partitions = [[2, [10, 0], G1, score(G1)], [3, [10, 0, 0], G2, score(G1)], 
                  [4, [10, 0, 0, 0]], G3, score(G1)]
    sol = recurse(partitions, 1, 1, lambda x: 100 * math.exp(2, 1 + .1 * x), 
                  lambda x: 100 * math.exp(x, -1))
    visualize(sol)
    print('score is')
    score(sol)

sanity_check()





