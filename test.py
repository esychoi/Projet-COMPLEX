from src import *

G = createGraphFromTxt('exempleinstance2.txt')
# draw_graph(G)

""" H = delete_node(G,0)
draw_graph(H) """

""" H = delete_list_node(G,[0,1])
draw_graph(H) """

""" d = degrees(G)
print(d) """

""" maxd = max_degree(G)
print(maxd) """

""" draw_graph(generate_random_graph(25,0.3)) """

""" print(G.edges)
print(algo_couplage(G)) """

""" print(algo_glouton(G)) """

""" print(branch(G))
print(branch2(G))
draw_graph(G) """

print(branch(G))
print(branch3(G))
draw_graph(G)
