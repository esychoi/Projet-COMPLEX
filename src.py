import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy


### PARTIE 1 : CREATION DE GRAPHE A PARTIR D'UN FICHIER TEXTE

def parser(filepath:str) -> nx.Graph():
    """function create graph from text file
    this function can only manage ints and tuples of ints
    format:
    '
    Nombre de sommets
    0
    Sommets
    0
    Nombre d aretes
    0
    Aretes
    0 1
    '
    Parameters
    ----------
    filepath : str
        path location of file
    Returns
    -------
    nx.Graph()
        graph containing nodes identified with ints in 'sommets' section and nodes identified with tuple of ints in 'aretes'
    """    
    
    with open(filepath) as f:
        lines = f.readlines()
    
    state = 0
    sommets = []
    aretes = []
    for line in lines:
        #print(line)
        #print(state)
        if state == 0:
            if 'Nombre de sommets' in line:
                state += 1
        elif state ==  1:
            if 'Sommets' in line:
                state += 1
        elif state ==  2:
            if 'Nombre d aretes' in line:
                state += 1
            else:
                sommets.append(int(line))
        elif state ==  3:
            if 'Aretes' in line:
                state += 1
        elif state ==  4:
            aretes.append(tuple([int(s) for s in line.split() if s.isdigit()]))
    G = nx.Graph()
    G.add_nodes_from(sommets)
    G.add_edges_from(aretes)
    return G

def draw_graph(G):
    """
    graph -> NoneType
    dessine le graphe G
    """
    nx.draw(G, with_labels=True)
    plt.show()

### PARTIE 2 : GRAPHE

# 2.1.1

def delete_node(G,v):
    """
    graph * node -> graph
    retoune un nouveau graphe obtenu à partir de G en supprimant le sommet v
    """
    newG = G.copy()
    newG.remove_node(v)
    return newG

# 2.1.2

def delete_list_node(G,s_v):
    """
    graph * list(node) -> graph
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
    """
    list((node,int)) -> (node,int)
    retourne le couple (n,i) de E où n est le sommet de plus grand degré i
    """
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
    G_tmp = G.copy()
    C = set()
    E = np.copy(G_tmp.edges)
    D = np.copy(G_tmp.degree)

    while len(E) > 0:
        v = max_degree_list(D)[0]
        C.add(v)
        G_tmp = delete_node(G_tmp,v)
        D = list(G_tmp.degree)
        E = list(G_tmp.edges)
    return C


### PARTIE 4

# 4.1

def branch(G):
    """
    graph -> set(nodes)
    retourne une couverture optimale de G en parcourant tout l'arbre d'énumération
    """
    C = set(G.nodes) # couverture la plus grande, que l'on va ameliorer en parcourant l'arbre d'enumeration
    pile = [(G,set())] # on commence avec le graphe initial et un ensemble vide
    #cpt = 0

    while(len(pile) > 0):
        p = pile.pop(0) # un noeud de l'arbre
        G_tmp = p[0]
        C_tmp = p[1]
        print(C_tmp)
        print(G_tmp.edges)
        draw_graph(G_tmp)

        if len(list(G_tmp.edges)) > 0:
            #cpt += 1
            e = list(G_tmp.edges)[0] # on prend une arete {u,v} de G_tmp
            C_tmp1 = C_tmp.copy()
            C_tmp1.add(e[0])
            C_tmp2 = C_tmp.copy()
            C_tmp2.add(e[1])
            pile = [(delete_node(G_tmp,e[0]),C_tmp1)] + pile    # branchement sur u
            pile = [(delete_node(G_tmp,e[1]),C_tmp2)] + pile    # branchement sur v
        else:
            if len(C) > len(C_tmp):
                C = C_tmp
    #print(cpt)
    return C

# 4.2

def couplage(G):
    """
    graph -> set(nodes)
    retourne un couplage de G
    """
    E = list(G.edges)
    G_tmp = G
    M = set()

    while E != []:
        M.add(E[0])
        G_tmp = delete_list_node(G_tmp,[E[0][0],E[0][1]])
        E = list(G_tmp.edges)
    return M

def calcul_bornes(G):
    """
    graph -> float
    retourne max(b1,b2,b3) avec b1, b2 et b3 définis comme dans le sujet
    """
    delta = max_degree(G)[1]
    n = len(G.nodes)
    m = len(G.edges)
    b1 = np.ceil(m/delta)
    b2 = len(couplage(G))
    b3 = (2*n-1 - np.sqrt((2*n-1)**2-8*m))/2
    return max([b1,b2,b3])

def branch2(G, glouton=False):
    """
    graph (* bool) -> set(nodes)
    retourne une couverture de G avec un algorithme de branch and bound
    """
    C = set(G.nodes) # couverture la plus grande, que l'on va ameliorer en parcourant l'arbre d'enumeration
    E = np.copy(G.nodes)
    pile = [(G,set())] # on commence avec le graphe initial et un ensemble vide
    borne_max = len(C)
    # nb_noeuds_visite = 0
    #cpt = 0

    while(len(pile) > 0):
        p = pile.pop(0) # un noeud de l'arbre
        G_tmp = p[0]
        C_tmp = p[1]
        
        
        if max_degree(G_tmp)[1] == 0:
            if len(C) > len(C_tmp):
                    C = C_tmp
        
        else:
            if calcul_bornes(G_tmp) < borne_max:
                # nb_noeuds_visite += 1
                if glouton:
                    #essai = len(C) + np.max([len(algo_couplage(G_tmp)), len(algo_glouton(G_tmp))])
                    essai = len(algo_glouton(G_tmp))
                else:
                    essai = len(algo_couplage(G_tmp)) 
                
                if essai < borne_max:
                    borne_max = essai


                if len(list(G_tmp.edges)) > 0:
                    #cpt += 1
                    e = list(G_tmp.edges)[0] # on prend une arete de G_tmp
                    C_tmp1 = C_tmp.copy()
                    C_tmp1.add(e[0])
                    C_tmp2 = C_tmp.copy()
                    C_tmp2.add(e[1])
                    pile = pile + [(delete_node(G_tmp,e[0]),C_tmp1)]
                    pile = pile + [(delete_node(G_tmp,e[1]),C_tmp2)]
                else:
                    if len(C) > len(C_tmp):
                        C = C_tmp
    #print(cpt)
    return C

# 4.3

def branch3(G, glouton=False):
    """
    graph (* bool) -> set(nodes)
    retourne une couverture de G avec un algorithme de branch and bound amélioré
    """
    C = set(G.nodes) # couverture la plus grande, que l'on va ameliorer en parcourant l'arbre d'enumeration
    E = np.copy(G.nodes)
    pile = [(G,set())] # on commence avec le graphe initial et un ensemble vide
    borne_max = len(C)
    # nb_noeuds_visite = 0
    #cpt = 0

    while(len(pile) > 0):
        p = pile.pop(0) # un noeud de l'arbre
        G_tmp = p[0]
        C_tmp = p[1]
        
        

        if max_degree(G_tmp)[1] == 0:
            if len(C) > len(C_tmp):
                    C = C_tmp
        
        else:
            if calcul_bornes(G_tmp) < borne_max:
                # nb_noeuds_visite += 1
                if glouton:
                    #essai = len(C) + np.max([len(algo_couplage(G_tmp)), len(algo_glouton(G_tmp))])
                    essai = len(C) + len(algo_glouton(G_tmp))
                else:
                    essai = len(algo_couplage(G_tmp)) + len(C)
                
                if essai < borne_max:
                    borne_max = essai


                if len(list(G_tmp.edges)) > 0:
                    #cpt += 1
                    
                    e = list(G_tmp.edges)[0] # on prend une arete {u,v} de G_tmp

                    C_tmp1 = C_tmp.copy()
                    C_tmp1.add(e[0])

                    C_tmp2 = C_tmp.copy()
                    C_tmp2.add(e[1])
                    """print('---')
                    print(G_tmp.nodes)
                    print(e[0])
                    print(C_tmp1)"""
                    voisins = G_tmp[e[0]]
                    C_tmp2 = C_tmp2.union(set(voisins))
                    """print(C_tmp2)
                    draw_graph(G_tmp)"""

                    pile = [(delete_node(G_tmp,e[0]),C_tmp1)] + pile #branche 1
                    if len((delete_list_node(G_tmp,[e[0],e[1]]+list(voisins))).nodes) != 0:
                        pile = [(delete_list_node(G_tmp,[e[0],e[1]]+list(voisins)),C_tmp2)] + pile #branche 2
                    else :
                        if len(C) > len(C_tmp2):
                            C = C_tmp2

                    #print(len(pile))
                else:
                    if len(C) > len(C_tmp):
                        C = C_tmp
    #print(cpt)
    return C

def branch32(G, glouton=False):
    """
    graph (* bool) -> set(nodes)
    retourne une couverture de G avec un algorithme de branch and bound amélioré
    """
    C = set(G.nodes) # couverture la plus grande, que l'on va ameliorer en parcourant l'arbre d'enumeration
    E = np.copy(G.nodes)
    pile = [(G,set())] # on commence avec le graphe initial et un ensemble vide
    borne_max = len(C)
    # nb_noeuds_visite = 0
    #cpt = 0

    while(len(pile) > 0):
        p = pile.pop(0) # un noeud de l'arbre
        G_tmp = p[0]
        C_tmp = p[1]
        
        

        if max_degree(G_tmp)[1] == 0:
            if len(C) > len(C_tmp):
                    C = C_tmp
        
        else:
            if calcul_bornes(G_tmp) < borne_max:
                # nb_noeuds_visite += 1
                if glouton:
                    #essai = len(C) + np.max([len(algo_couplage(G_tmp)), len(algo_glouton(G_tmp))])
                    essai = len(C) + len(algo_glouton(G_tmp))
                else:
                    essai = len(algo_couplage(G_tmp)) + len(C)
                
                if essai < borne_max:
                    borne_max = essai


                if len(list(G_tmp.edges)) > 0:
                    #cpt += 1
                    u = max_degree(G_tmp)[0] # sommet de degre max
                    v = list(G[u])[0] # sommet voisin de u
                    e = (u,v)# on a une arete {u,v} de G_tmp

                    C_tmp1 = C_tmp.copy()
                    C_tmp1.add(e[0])

                    C_tmp2 = C_tmp.copy()
                    C_tmp2.add(e[1])
                    """print('---')
                    print(G_tmp.nodes)
                    print(e[0])
                    print(C_tmp1)"""
                    voisins = G_tmp[e[0]]
                    C_tmp2 = C_tmp2.union(set(voisins))
                    """print(C_tmp2)
                    draw_graph(G_tmp)"""

                    pile = [(delete_node(G_tmp,e[0]),C_tmp1)] + pile #branche 1
                    if len((delete_list_node(G_tmp,[e[0],e[1]]+list(voisins))).nodes) != 0:
                        pile = [(delete_list_node(G_tmp,[e[0],e[1]]+list(voisins)),C_tmp2)] + pile #branche 2
                    else :
                        if len(C) > len(C_tmp2):
                            C = C_tmp2

                    #print(len(pile))
                else:
                    if len(C) > len(C_tmp):
                        C = C_tmp
    #print(cpt)
    return C
