p# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



#%%
# Messing around with python a bit - this is not CAVE

import numpy as np
import matplotlib.pyplot as plt

# example data
x = np.arange(0, 20, 0.1)
y = np.sqrt(x)

fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()

#%%

import numpy as np
import time

x = np.random.random_integers(15, size = [2000000,1])
y = np.random.random_integers(15, size = [2000000,1])

tic = time.time()

Dot1 = np.dot(x.T,y)

toc = time.time()
elapsed1 = toc - tic

Dot2 = 0

tic = time.time()

for i in range(len(x)):
    Dot2 += x[i]*y[i]        

toc = time.time()
elapsed2 = toc - tic

print(elapsed1)
print(elapsed2)

#%%

m = np.random.randint(25)
n = np.random.randint(25)
p = np.random.randint(25)

X = np.random.random_integers(15, size = [m,n])
Y = np.random.random_integers(15, size = [n,p])

#%%

# Retrieved from 
# https://docs.scipy.org/doc/scipy/reference/sparse.html

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from numpy.linalg import solve, norm
from numpy.random import rand
import time

A = lil_matrix((10000, 10000)) #List of Lists format
A[0, :1000] = rand(1000) #Set the first 100 columns in row 0 to random numbers
A[1, 1000:2000] = A[0, :1000] #Set the 100-199 columns in row 1 to the same nums
A.setdiag(rand(10000)) #Set diagonal of sparse matrix to random numbers


A = A.tocsr() #Set matrix to a compressed sparse row format
b = rand(10000) #make random b vector

tic = time.time()

x3 = spsolve(A,b) #Solve

elapsed3 = time.time()-tic

#########                    
                    
tic = time.time()

x4 = solve(A.toarray(),b)
                    
elapsed4 = time.time()-tic

#########
                    
A_ = A.toarray()
                    
tic = time.time()

x5 = solve(A_,b)
                    
elapsed5 = time.time()-tic

#########

print(elapsed3)
print(elapsed4)
print(elapsed5)                    
#%%

# Optimize usiung scipy
# Portions of the L-shaped algorithm example from Papavasiliou

import numpy as np
from scipy.optimize import linprog
from scipy.optimize import linprog_verbose_callback

c1 = np.array([100, 150])
A_ub1 = np.array([[1, 1], [-1,0], [0,-1]])
b_ub1 = np.array([120, -40, -20])

c = np.array([-24, -28])

A_ub = np.array([[6, 10], [8, 5],[1,0],[0,1]])
b_ub = np.array([2400, 1600, 500, 100])

res = linprog(c, A_ub, b_ub, options={"disp": True})
test = linprog_verbose_callback(res.x, "tableau")

c2 = np.array([100, 150, 1])
A_ub2 = np.array([[1, 1, 0],[-83.52, -180.48, -1], [-211.2, 0, -1]])
b_ub2 = np.array([120, 520, 1584])
x1_bounds = (40, None)
x2_bounds = (20, None)
x3_bounds = (None, None)

c3 = np.array([-28, -32])
Au3 = np.array([[6, 10], [8, 5],[1,0],[0,1]])
bu3 = np.array([2400, 6400,300,300])

c3 = np.array([-24, -28])
Au3 = np.array([[6, 10], [8, 5],[1,0],[0,1]])
bu3 = np.array([2400, 6400,500,100])

c = np.array([0,0,1,1,1,1,1,1,1,1,1,1,1,1])

A = np.array([[3,2,1,0,0,0,0,0,-1,0,0,0,0,0],
              [2,5,0,1,0,0,0,0,0,-1,0,0,0,0],
              [-1,0,0,0,-1,0,0,0,0,0,1,0,0,0],
              [1,0,0,0,0,1,0,0,0,0,0,-1,0,0],
              [0,-1,0,0,0,0,-1,0,0,0,0,0,1,0],
              [0,1,0,0,0,0,0,1,0,0,0,0,0,-1]])

b = np.array([0,0,-4.8,6,-6.4,8])

#%%

# Binomial test

import numpy as np
import scipy as sp
from scipy.stats import binom

n, p = 8, .5
P = binom.pmf(np.arange(n+1),n,p)

# The below does the same thing, but the above is much more compact and pythonic

#P = np.zeros(n+1)
#
#for k in np.arange(9):
#    P[k] = binom.pmf(k, n, p)
#
#print(P)   

#x = np.arange(binom.ppf(0.01, n, p), binom.ppf(n.p))
#test = binom.pmf(x, n, p)

#%% Maximum Marginal Return

import numpy as np
import scipy as sp

num_weap = int(input("Please enter the number of weapons: "))
num_targ = int(input("Please enter the number of targets: "))
p_k = float(input("Please enter the single-shot probability of kill for weapon-target pairs: "))

# np.where(test > 40) #For future use?
# int(np.where(test == np.max(test))[0]) # also very useful

##### Homogeneous weapons and targets #####

# Define target values
#T = 200*np.ones(num_targ)
T = np.random.randint(100, 750, 8)
P = 0.5*np.ones(num_weap)
weap_ctr = num_weap
Value = 0

while weap_ctr > 0:
    # find target with the maximum value
    if np.shape(np.where(T == np.max(T))[0])[0] > 0:
        #break ties by choosing the first in the array, 
        curr_tar_index = min(np.where(T==np.max(T))[0])
    else: 
        curr_tar_index = np.where(T == np.max(T))[0]
    # find weapon with greatest probability of kill 
    if np.shape(np.where(P == np.max(P))[0])[0] > 0:
        #break ties by choosing the first in the array, 
        curr_weap_index = min(np.where(P==np.max(P))[0])
    else: 
        curr_weap_index = np.where(P == np.max(P))[0]
    
    #expected value of target "damage"
    curr_value = T[curr_tar_index]-T[curr_tar_index]*P[curr_weap_index]
    print(curr_value)
    P[curr_weap_index] = 0
    T[curr_tar_index] = curr_value
    Value += curr_value
    weap_ctr -= 1
    
#%% More MMR implementations

from operator import itemgetter

from numba import jit
import numpy as np


def mmr_1(p_kill, targets, verbose=True):
    """MMR algorithm initial translation."""
    n_weapons = len(p_kill)
    value = 0
    allocation = np.zeros_like(targets)
    if verbose:
        print(value)
        print(allocation)
        print(targets)
    while n_weapons > 0:
        target_idx = np.argmax(targets)
        weapon_idx = np.argmax(p_kill)
        temp_value = targets[target_idx] - targets[target_idx] * p_kill[weapon_idx]
        p_kill[weapon_idx] = 0
        targets[target_idx] -= temp_value
        allocation[target_idx] += 1
        value += temp_value
        n_weapons -= 1
        if verbose:
            print(value)
            print(targets)
            print(allocation)


def mmr_2(p_kill, targets, verbose=True):
    """MMR algorithm with sorting/looping on p_kill."""
    value = 0
    allocation = np.zeros_like(targets)
    p_kill.sort()
    p_kill = p_kill[::-1]
    if verbose:
        print(value)
        print(allocation)
        print(targets)
    for p in p_kill:
        idx = np.argmax(targets)
        target = targets[idx]
        temp_value = target - target * p
        targets[idx] -= temp_value
        allocation[idx] += 1
        value += temp_value
        if verbose:
            print(value)
            print(targets)
            print(allocation)
            

def mmr_3(p_kill, targets, verbose=True):
    """MMR algorithm with allocation by index rather than count."""
    val = 0
    allocation = np.zeros_like(p_kill)
    p_kill.sort()
    p_kill = p_kill[::-1]
    if verbose:
        print(val)
        print(allocation)
        print(targets)
    for p_idx, p in enumerate(p_kill):
        target_idx = np.argmax(targets)
        target = targets[target_idx]
        temp_val = target - target * p
        targets[target_idx] -= temp_val
        allocation[p_idx] = target_idx
        val += temp_val
        if verbose:
            print(val)
            print(targets)
            print(allocation)
            

def mmr_4(p_kill, targets, verbose=True):
    """MMR algorithm with breaks on target consumption and NaN use."""
    val = 0
    allocation = np.repeat(np.NaN, repeats=p_kill.size)
    p_kill.sort()
    p_kill = p_kill[::-1]
    if verbose:
        print(val)
        print(allocation)
        print(targets)
    for p_idx, p in enumerate(p_kill):
        target_idx = np.argmax(targets)
        target = targets[target_idx]
        if not target > 0:
            break
        temp_val = target - target * p
        targets[target_idx] -= temp_val
        allocation[p_idx] = target_idx
        val += temp_val
        if verbose:
            print(val)
            print(targets)
            print(allocation)


def mmr_5(p_kill, targets, verbose=True):
    """No-numpy MMR!"""
    val = 0
    allocation = [None] * len(p_kill)
    p_kill = sorted(p_kill, reverse=True)
    if verbose:
        print(val)
        print(allocation)
        print(targets)
    for p_idx, p in enumerate(p_kill):
        target_idx, target = max(enumerate(targets), key=itemgetter(1))
        if not target > 0:
            break
        temp_val = target - target * p
        targets[target_idx] -= temp_val
        allocation[p_idx] = target_idx
        val += temp_val
        if verbose:
            print(val)
            print(targets)
            print(allocation)


np.random.seed(10)
print("Small Sample Size:")            
for mmr in [mmr_1, mmr_2, mmr_3, mmr_4, mmr_5]:
    print(mmr.__name__)
    p_kill = np.random.random(10)
    targets = np.random.randint(1, 1000, size=15)
    %timeit mmr(p_kill=p_kill, targets=targets, verbose=False)
    
print("Medium Sample Size:") 
for mmr in [mmr_1, mmr_2, mmr_3, mmr_4, mmr_5]:
    print(mmr.__name__)
    p_kill = np.random.random(100)
    targets = np.random.randint(1, 1000, size=150)
    %timeit mmr(p_kill=p_kill, targets=targets, verbose=False)
    
print("Large Sample Size:") 
for mmr in [mmr_1, mmr_2, mmr_3, mmr_4, mmr_5]:
    print(mmr.__name__)
    p_kill = np.random.random(1000)
    targets = np.random.randint(1, 10000, size=15000)
    %timeit mmr(p_kill=p_kill, targets=targets, verbose=False)

#%% Search Tree Algorithm

# Found at https://www.python.org/doc/essays/graphs/

# Initialize
# Define the network in dictionary data structure
net = {'A': ['B', 'C'],
       'B': ['C', 'D'],
       'C': ['F'],
       'D': ['E', 'G'],
       'E': ['B', 'C','G'],
       'F': ['G']}

# Find a path from 'start' to 'end'
def find_path(graph, start, end, path=[]):
        path = path + [start]
        if start == end:
            return path
        if not start in graph: #changed from graph.has_key(start)
            return None
        for node in graph[start]:
            if node not in path:
                newpath = find_path(graph, node, end, path)
                if newpath: return newpath
        return None
    
# Find all paths (without cycles)
 def find_all_paths(graph, start, end, path=[]):
        path = path + [start]
        if start == end:
            return [path]
        if not start in graph: #changed from graph.has_key(start)
            return []
        paths = []
        for node in graph[start]:
            if node not in path:
                newpaths = find_all_paths(graph, node, end, path)
                for newpath in newpaths:
                    paths.append(newpath)
        return paths
    
# Find shortest path - based on number of arcs
# Should be fairly simple to add arc lengths to find the min/max flow
 def find_shortest_path(graph, start, end, path=[]):
        path = path + [start]
        if start == end:
            return path
        if not start in graph: #changed from graph.has_key(start)
            return None
        shortest = None
        for node in graph[start]:
            if node not in path:
                newpath = find_shortest_path(graph, node, end, path)
                if newpath:
                    if not shortest or len(newpath) < len(shortest):
                        shortest = newpath
        return shortest


#%% 
# Search tree algorithm from Network Flows by Ahuja, Magnanti, and Orlin pg 74

import numpy as np

net =   {'A': ['B', 'C'],
         'B': ['C', 'D', 'E'],
         'C': ['D'],
         'D': ['F'],
         'E': ['D', 'F'],
         'F': ['G'],
         'G': []}

def search_tree(graph, start, path=[]):
    counter = 1 #helps define order of traversal (next =1 from textbook algorithm)
    marked = set(start) #set of marked nodes - starts with the beginning
    order = {start:counter} #dictionary of order of traversal for nodes
    pred = {start:[]} #dictionary of predecessor nodes for every visited node
    list_ = set(start) #initial list is simply the set of the starting node
    
    while len(list_) > 0:
        for node in list_:
            print("Node: " + node)
            for Adj in (set(graph.get(node,[]))-marked):
                print("Incident node: " + Adj)
                marked = marked | set(Adj)
                print("Marked set: " + str(marked))
                pred[Adj] = node
                counter += 1
                print("Counter = " + str(counter))
                order[Adj] = counter
                list_ = list_ | set(Adj)
            list_ = list_ - set(node)
            print(list_)
    return(order, pred)   
    

#def search_tree(graph, start, path = []):
#    #initialize
##    marked = {}
##    order = {}
##    marked[start] = 1
#    pred={start: []}
##    nxt = 1
##    order[start] = 1
#    List = {start}
#    N = set(graph.keys())
#    
#    while len(List) > 0:
#        print(List)
#        for node in List:
#            print("List is: " + str(List))
#            print("Current node is: " + node)
#            if len(graph.get(node,[])) == 0:
#                print("Algorithm has completed")
#                List.remove(node)
#            elif len(graph.get(node,[])) > 0:
#                for Adj in graph.get(node,[]):                    
#                    print("Adjacent node is: " + Adj)
##                    marked[Adj] = 1
#                    pred[Adj] = node
##                    nxt += 1
##                    order[Adj] = nxt
#                    List.append(Adj)
##                    All_Adj.remove(Adj)
##                    graph[node] = All_Adj
#                print("The network is currently: " + str(graph))
#                print("List is currently: " + str(List))
##                    print(marked)
#                print("The predeccessors are: " + str(pred))
#                List.remove(node)
#                print("Node " + node + " has been removed")
#            else:
#                List.remove(node)
#                print("You have removed node: " + node)

#%% 
# Topological ordering algorithm from Network Flows by Ahuja, Magnanti, and Orlin pg 79

import numpy as np
from collections import defaultdict
from collections import Counter


net =   {'A': ['B', 'C'],
         'B': ['C', 'D', 'E'],
         'C': ['D'],
         'D': ['F'],
         'E': ['D', 'F'],
         'F': ['G'],
         'G': []}

N = set(net.keys())

def indegree(graph):
    indegree = Counter ()
    N = set(graph.keys())
    for node in N:
        for next_node in graph.get(node,[]):
            indegree[next_node] += 1
    return(indegree)
           
def topo_order(graph):
    order = []
    full_len = len(graph.keys())
    while len(order) < full_len:
        N = set(graph.keys())
        I = indegree(graph)
        if len(N) == len(set(I)):
            order = np.zeros(full_len)
        else:
            for node in N:
                if I.get(node) is None:
                    del graph[node]
                    order += node
    return(order)
            
                   

############ Setup Code: Page 77 onwards, "Topological Ordering" ##############

# Figure 3.6(a) Acyclic Network
graph = {'A': {'D'},
         'B': {'A', 'D', 'E'},
         'C': set(),
         'D': {'C'},
         'E': {'C', 'D'}}

# Figure 3.6(c) and (d) Topological Ordering of 3.6(a)
correct_topo_orders = [['B', 'A', 'E', 'D', 'C'],
                       ['B', 'E', 'A', 'D', 'C']]

# Figure 3.7 Cyclic Network
cyclic = {'A': {'B'},
          'B': {'C'},
          'C': {'A'}}

# Note that both of these examples return an empty list for a cyclic network

##### Version 1 ################################################################
def find_topological_order(graph, *, sort_dictitem_tuple=lambda x: x[1],
                           copy_args=True):
    if copy_args:
        graph = graph.copy() # to avoid clobbering the input graph 
    order_length = len(graph) # has to be cached otherwise it changes
    order = []
    while len(order) < order_length:
        n_degrees = {}
        for node in graph:
            if node not in n_degrees:
                n_degrees[node] = 0
            for arc in graph[node]:
                if arc not in n_degrees:
                    n_degrees[arc] = 1
                else:
                    n_degrees[arc] += 1
            if not order and min(n_degrees.values()) > 0:
                return order # return an empty list early if cyclic
        node, n_degree = min(n_degrees.items(), key=sort_dictitem_tuple)
        order.append(node)
        del graph[node]
    return order

print(find_topological_order(graph)) 
print(find_topological_order(cyclic))



##### Version 2 ################################################################
from collections import Counter, defaultdict
from itertools import chain


def find_topological_order(graph, *, sort_dictitem_tuple=lambda t: t[1],
                           create_zero=lambda: 0, copy_args=True):
    if copy_args:
        graph = graph.copy()
    order_length = len(graph)
    order = []
    while len(order) < order_length: # the next two lines are pretty gnarly
        n_degrees = defaultdict(create_zero, **Counter(chain(*graph.values())))
        _ = [n_degrees[node] for node in graph if node not in n_degrees]
        if not order and min(n_degrees.values()) > 0:
            return order
        node, n_degree = min(n_degrees.items(), key=sort_dictitem_tuple)
        order.append(node)
        del graph[node]
    return order

print(find_topological_order(graph)) 
print(find_topological_order(cyclic))
%timeit find_topological_order(graph)

#%% 
############################ Dijkstra's Algorithm #############################

def dijkstra(graph):
    S = {}
    S_hat = set(graph.keys())
    for key in dofd_net.keys():
        dofd_net[key]['eye'] = np.inf
        dofd_net[key]['ess'] = 0
        dofd_net[key]['pred'] = 0
    

############################ Network from page 110 ############################
dofd_net =   {'A': {'B': {'weight': 6}, 'C':{'weight': 4}}, 
         'B': {'C': {'weight': 2}, 'D':{'weight': 2}},
         'C': {'D': {'weight': 1}, 'E':{'weight': 2}},
         'D': {'F': {'weight': 7}}, 
         'E': {'D': {'weight': 4}, 'F':{'weight': 7}},
         'F': {}}

#%%

############################ CAVE Algorithm ###################################

def CAVE(*,*,*,*):
    #CAVE Algorithm here
    
