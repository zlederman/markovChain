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
import numpy as np

np.set_printoptions(threshold=np.inf)
dic = defaultdict(list)
class Graph:
    def __init__(self,V,root):
        self.V = V
        self.nodes = []
        self.root = root
        self.add_node(root)
    def add_node(self, node):
        if self.nodes is None:
            self.nodes = [node]
        else:

            self.nodes.append(node)
    def get_node(self, word):
        for i in range(len(self.nodes)):
            if word == self.nodes[i].word:
                return self.nodes[i]
        print("word not found")
        return None

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

def tokenize(corpus):
    arr = corpus.split(' ')
    arr = [clean_element(word) for word in arr]
    for i in range(len(arr) - 1):

        if arr[i] not in dic:
            dic[arr[i]] = [arr[i+1]]
        else:
            dic[arr[i]].append(arr[i+1])

def get_probabilities():
    for k, v in dic.items():
        new_value = calc_probabilities(v)
        new_value.sort(key= lambda x: x[1])
        dic[k] = new_value

def is_in(arr,wrd):
    if len(arr) == 0:
        return True
    for tup in arr:

        if tup[0] == wrd:
            return False
    return True

def calc_probabilities(arr):
    total = len(arr)
    p_arr = []
    for i in range(total):
        if is_in(p_arr, arr[i]):
            p_arr.append((arr[i], arr.count(arr[i])/total))
    return p_arr

# def get_elements(word,max_depth, max_leaves,tups):
#
#     if max_depth == 0:
#         return
#     else:
#         temp_leaves = max_leaves
#         if len(dic[word]) < max_leaves:
#             temp_leaves = len(dic[word])
#         temp = dic[word][0:temp_leaves]
#
#         tups.append({word:temp})
#     for word in temp:
#         get_elements(word[0], max_depth - 1, max_leaves, tups)

def make_node(word):

    return node(word)
#get_elements(word,3, 3, G.add(node(word[0])))
def get_elements(max_depth, max_leaves,node):

    if max_depth == 0:
        return
    else:

        word = node.word
        num_leaves = max_leaves
        prob_array = dic[word]
        if len(dic[word]) < max_leaves:
            num_leaves = len(prob_array)


        for i in range(num_leaves):
            next_word = prob_array[i]
            token, prob = next_word[0], next_word[1]
            token_node = make_node(token)
            node.add_connection(prob, token_node)
            get_elements(max_depth - 1, max_leaves, token_node)
    return
# def num_nodes(tups):
#     for i in range(len(tups)):
#         for k,v in tups[i].items():

def build_graph(graph, tups):
    for i in range(len(tups)):
        k, v = next(iter(tups[i].items()))
        n = node(k)
        # print(v)
        for i in range(len(v)):
            n.add_connection(v[i][1],v[i][0])
        graph.add_nodes(n)


def build_node_list(G, node,max_depth):
    if max_depth == 0:
        return
    for n in node.get_connections():

        G.add_node(n)
        build_node_list(G, n, max_depth - 1)
    return
'''

dic = {
        word : [(str, float64),(str, float64),...,(str, float64)]
        word : [(str, float64),(str, float64),...,(str, float64)]
        word : [(str, float64),(str, float64),...,(str, float64)]    
        word : [(str, float64),(str, float64),...,(str, float64)]
        ...
        }
G = {
        node : {name : word, 
                connections: {
                             word : edge-weight
                             word : edge-weight
                            }
                }          
            
                }
                
tups = [ {word: [leaf leaf leaf] }
get_files(s,e) -> gets files, then loads elements into an array
to be tokenized and cleaned of any non-alpha numeric tokens

get_probabilities()-> calls calc_prob(), which calculates the probability of an element 
based off of it's occurance in the dict, makes it into a (str, float64)

get_elements()-> performs a BFS on each array from a given key to
thus appends  

'''

def main():
    word_inp = input("String: ")
    max_depth = 5
    get_files(0, 30)
    get_probabilities()


    tups = []
    root = node(word_inp)
    word_graph = Graph(0,root)

    get_elements(max_depth,3,word_graph.get_node(word_inp))
    # get_elements('are',10,10,tups)
    build_node_list(word_graph, root, max_depth)
    # graph = Graph(0)
    # build_graph(graph,tups)

    G = nx.Graph()
    for n in word_graph.nodes:
        for k, v in n.get_connections().items():
            G.add_edge(n.word,k.word,weight=v)
    e_sizes = []
    for i in range(1,10):
        e_sizes.append([(u, v) for (u, v, d) in G.edges(data=True) if (i/10) > d["weight"] > ((i-1)/10)])



    colors = [(1,0,0),(0,0,1),(0,1,0),(1,128/255,0),(1,1,51/255),(1,51/255,255/255),(0,0,0),(255/255,51/255,153/255),(102/255,0,51/255),(0,102/255,204/255),(0,1,1)]
    pos = nx.spring_layout(G)  # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=10,node_color='red')
    plt.axis("off")

    # plt.savefig("./graph.png", dpi=400)

    # edges
    for i in range(0,9):

        nx.draw_networkx_edges(G, pos, edgelist=e_sizes[i], width=.4, edge_color=colors[i], label=str(i/10))


    nx.draw_networkx_labels(G, pos, font_size=3, font_family="sans-serif")
    plt.show()
    next_word = word_inp
    max_iter = max_depth
    while max_iter:
        u = word_graph.get_node(next_word)
        arr = []
        for k,v in u.get_connections().items():
            arr.append({k.word})
        p = input(f"{arr}")
        next_word = p
        root = node(next_word)
        word_graph = Graph(0, root)

        get_elements(max_depth, 3, word_graph.get_node(next_word))
        # get_elements('are',10,10,tups)
        build_node_list(word_graph, root, max_depth)
        G = nx.Graph()
        for n in word_graph.nodes:
            for k, v in n.get_connections().items():
                G.add_edge(n.word, k.word, weight=v)
        e_sizes = []
        for i in range(1, 10):
            e_sizes.append([(u, v) for (u, v, d) in G.edges(data=True) if (i / 10) > d["weight"] > ((i - 1) / 10)])

        colors = [(1, 0, 0), (0, 0, 1), (0, 1, 0), (1, 128 / 255, 0), (1, 1, 51 / 255), (1, 51 / 255, 255 / 255),
                  (0, 0, 0), (255 / 255, 51 / 255, 153 / 255), (102 / 255, 0, 51 / 255), (0, 102 / 255, 204 / 255),
                  (0, 1, 1)]
        pos = nx.spring_layout(G)  # positions for all nodes

        # nodes
        nx.draw_networkx_nodes(G, pos, node_size=10, node_color='red')

        # edges
        for i in range(0, 9):
            nx.draw_networkx_edges(G, pos, edgelist=e_sizes[i], width=.4, edge_color=colors[i], label=str(i / 10))

        nx.draw_networkx_labels(G, pos, font_size=3, font_family="sans-serif")
        plt.axis("off")

        # plt.savefig("./graph.png", dpi=400)
        plt.show()

        max_iter = max_iter - 1

    # # labels




    plt.axis("off")

    # plt.savefig("./graph.png", dpi=400)
    plt.show()



main()
# Press the green button in the gutter to run the script.


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
