# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import os
import sys
import pandas as pd
import scipy
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx


dic = defaultdict(list)
class Graph:
    def __init__(self,V):
        self.V = V
        self.nodes = []
    def add_nodes(self, node):
        if self.nodes is None:
            self.nodes = [node]
        else:

            self.nodes.append(node)

class node:

    def __init__(self,word):
        self.word = word
        self.connections = {}

    def add_connection(self, edge, word):

        self.connections[word] = edge
    def get_connections(self):
        return self.connections

#TODO implement graph structure so i can get the adjacency matrix

def get_files(start,end):
    for i in range(start,end):
        filename = 'unsup/' + str(i) + '_0.txt'
        with open(os.path.join(sys.path[0],filename)) as f:
            corpus = f.read()
            corpus.strip()
            tokenize(corpus)



def clean_element(word):
    return re.sub(r'\W+', '', word)

def tokenize(corp):
    arr = corp.split(' ')
    arr = [clean_element(word) for word in arr]
    for i in range(len(arr) - 1):

        if arr[i] not in dic:
            dic[arr[i]] = [arr[i+1]]
        else:
            dic[arr[i]].append(arr[i+1])

def clean_dic():
    for k, v in dic.items():

        new_value = get_ps(v)
        new_value.sort(key= lambda x: x[1])
        dic[k] = new_value

def is_in(arr,wrd):
    if len(arr) == 0:
        return True
    for tup in arr:

        if tup[0] == wrd:
            return False
    return True

def get_ps(arr):
    total = len(arr)
    p_arr = []
    for i in range(total):
        if is_in(p_arr, arr[i]):
            p_arr.append((arr[i], arr.count(arr[i])/total))
    return p_arr

def get_elements(word,max_depth, max_leaves,tups):

    if max_depth == 0:
        return
    else:
        temp_leaves = max_leaves
        if len(dic[word]) < max_leaves:
            temp_leaves = len(dic[word])
        temp = dic[word][0:temp_leaves]


        tups.append({word :temp})
    for word in temp:
        get_elements(word[0], max_depth - 1, max_leaves, tups)

# def num_nodes(tups):
#     for i in range(len(tups)):
#         for k,v in tups[i].items():

def build_graph(graph, tups):
    for i in range(len(tups)):
        k, v = next(iter(tups[i].items()))
        n = node(k)
        print(v)
        for i in range(len(v)):
            n.add_connection(v[i][1],v[i][0])
        graph.add_nodes(n)



def main():
    get_files(0, 10)
    clean_dic()

    tups = []
    get_elements('are',5,5,tups)

    graph = Graph(0)
    build_graph(graph,tups)

    G = nx.Graph()
    for n in graph.nodes:
        for k, v in n.get_connections().items():
            G.add_edge(n.word,k,weight=v)
    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 0.5]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 0.5]

    pos = nx.spring_layout(G)  # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=10,node_color='red')

    # edges
    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=.4)
    nx.draw_networkx_edges(
        G, pos, edgelist=esmall, width=.4, alpha=0.5, edge_color="b", style="dashed"
    )
    nx.draw_networkx_labels(G, pos, font_size=3, font_family="sans-serif")
    # labels


    plt.axis("off")

    plt.savefig("./graph.png", dpi=400)
    plt.show()





if __name__ == "__main__":
    main()
# Press the green button in the gutter to run the script.


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
