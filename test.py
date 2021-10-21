from src import *
from time import time

#G = parser('exempleinstance2.txt')
G = generate_random_graph(10,0.62)
#print(algo_couplage(G))
#print(algo_glouton(G))
print(branch(G))
print(branch2(G))
print(branch3(G))
print(branch32(G))
#draw_graph(G)

### TESTS 3.2 : ALGO_COUPLAGE ET ALGO_GLOUTON

""" N_max = 600
p = 1
print("N_max =", N_max)
print("p =", p)
for k in range(1,11):
    n = int(k*N_max/10)
    tn_generation = 0.0
    tn_couplage = 0.0
    tn_glouton = 0.0
    print("n =", n)
    for i in range(10):
        t1 = time()
        G = generate_random_graph(n,p)
        t2 = time()
        tn_generation += (t2-t1)
        #algo_couplage
        t1 = time()
        algo_couplage(G)
        t2 = time()
        tn_couplage += (t2-t1)
        
        #algo_glouton
        t1 = time()
        algo_glouton(G)
        t2 = time()
        tn_glouton += (t2-t1)
    # calcul de la moyenne des temps d'exécution
    tn_generation = tn_generation/10
    tn_couplage = tn_couplage/10
    tn_glouton = tn_glouton/10
    print("\tgeneration du graphe :", tn_generation)
    print("\t\t log :", np.log(tn_generation))
    print("\talgo_couplage :", tn_couplage)
    print("\t\t log :", np.log(tn_couplage))
    print("\talgo_glouton :", tn_glouton)
    print("\t\t log :", np.log(tn_glouton)) """