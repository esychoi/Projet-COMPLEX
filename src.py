import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy


### PARTIE 1 : CREATION DE GRAPHE A PARTIR D'UN FICHIER TEXTE

def draw_graph(G):
    """
    graph -> NoneType
    dessine le graphe G
    """
    nx.draw(G, with_labels=True)
    plt.show()

def extract_nodes(s):
    """
    string -> (int,int)
    s : chaine de caractere d'une arete
    retourne les noms des sommets de s
    """
    i,s1,s2 = 0,'',''

    while s[i] != ' ':
        s1 += s[i]
        i += 1
    i += 1
    while s[i] != '\n':
        s2 += s[i]
        i += 1
    return (int(s1),int(s2))

def createGraphFromTxt(filename, show=False):
    """
    str * bool (False by default) -> graph
    retourne le graphe décrit dans le fichier filename
    """
    try:
        f = open(filename, "r")
        nodes = [] #liste des sommets
        edges  = [] #liste des arêtes

        f.readline() # "Nombre de sommets\n"
        f.readline() # nb de sommets
        checkpoint = f.readline() # "Sommets\n", checkpoint initial
        line = f.readline() # premier sommet

        while(line):
            if line == "Nombre d aretes\n":
                checkpoint = "Aretes" #fin lecture de sommets, lecture des aretes
                f.readline() # nb d'aretes
                f.readline() #"Aretes\n"

            elif checkpoint == "Sommets\n" and str.isnumeric(line.replace("\n", "")):
                nodes.append(int(line))

            elif checkpoint == "Aretes":
                a = extract_nodes(line)
                edges.append(a)

            line = f.readline()

        #creation du graphe
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        if(show):
            draw_graph(graph)

        return graph

    finally:
        f.close()


### PARTIE 2 : GRAPHE

# 2.1.1

def delete_node(G,v):
    """
    graph * node
    retoune un nouveau graphe obtenu à partir de G en supprimant le sommet v
    """
    newG = G.copy()
    newG.remove_node(v)
    return newG

# 2.1.2

def delete_list_node(G,s_v):
    """
    graph * list(node)
    retoune un nouveau graphe obtenu à partir de G en supprimant les sommets de s_v
    """
    newG = G.copy()
    newG.remove_nodes_from(s_v)
    return newG

# 2.1.3

def degrees(G):
    """
    graph -> list((node,int))
    retourne la liste des couples (n,i) où n est un sommet et i son degré
    """
    return G.degree

def max_degree(G):
    """
    graph -> (node,int)
    retourne le couple (n,i) où n est le sommet de plus grand degré i
    """
    return max(G.degree,key=lambda x:x[1])

def max_degree_list(E):
    return max(E,key=lambda x:x[1])

# 2.2
def generate_random_graph(n,p):
    """
    int * float -> graph
    hyp : p € ]0,1[
    retourne le graphe dont les sommets sont {0,...,n} et où chaque arete {i,j}
        apparaît avec probabilité p
    """
    G = nx.Graph()
    G.add_nodes_from(np.arange(n))
    for i in range(n):
        for j in range(i+1,n):
            if np.random.rand() < p:
                G.add_edge(i,j)
    return G


### PARTIE 3 : METHODES APPROCHEES

# 3.2

def algo_couplage(G):
    """
    graph -> set(node)
    retourne une couverture de G
    """
    C = set()
    for e in G.edges:
        if (not e[0] in C) and (not e[1] in C):
            C.add(e[0])
            C.add(e[1])
    return C

def algo_glouton(G):
    """
    graph -> set(node)
    retourne une couverture de G
    """
    C = set()
    E = np.copy(G.edges)
    while len(E) > 0:
        v = max_degree_list(E)[0]
        C.add(v)
        E = E[np.where((E[:,0] != v) & (E[:,1] != v))]
    return C


### PARTIE 4

# 4.1

def branch(G):
    C = set(G.nodes) # couverture la plus grande, que l'on va ameliorer en parcourant l'arbre d'enumeration
    E = np.copy(G.nodes)
    pile = [(G,set())] # on commence avec le graphe initial et un ensemble vide

    while(len(pile) > 0):
        p = pile.pop(0) # un noeud de l'arbre
        G_tmp = p[0]
        C_tmp = p[1]

        if len(list(G_tmp.edges)) > 0:
            e = list(G_tmp.edges)[0] # on prend une arete de G_tmp
            C_tmp1 = C_tmp.copy()
            C_tmp1.add(e[0])
            C_tmp2 = C_tmp.copy()
            C_tmp2.add(e[1])
            pile = [(delete_node(G_tmp,e[0]),C_tmp1)] + pile
            pile = [(delete_node(G_tmp,e[1]),C_tmp2)] + pile
        else:
            if len(C) > len(C_tmp):
                C = C_tmp
    return C
