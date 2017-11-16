import networkx as nx

# for Graph visualisation
import matplotlib.pyplot as plt

def plot_graph(g):
    plt.subplot()
    nx.draw(g, with_labels=True)
    plt.show()

# the hand graph as shown in the kickoff slides
kickoff_hand = [("t0", "wrist"), ("i0", "wrist"), ("m0", "wrist"), ("r0", "wrist"), ("p0", "wrist"), ("i0", "m0"),
                ("m0", "r0"), ("r0", "p0"), ("t0", "t1"), ("t1", "t2"), ("i0", "i1"), ("i1", "i2"), ("i2", "i3"),
                ("m0", "m1"), ("m1", "m2"), ("m2", "m3"), ("r0", "r1"), ("r1", "r2"), ("r2", "r3"), ("p0", "p1"),
                ("p1", "p2"), ("p2", "p3")]

# create the hand graph from given edge list
G = nx.Graph(kickoff_hand)
print "Neighbours of i0:"
for x in nx.all_neighbors(G, "i0"): print x
#plot_graph(G)
print "\nAll nodes in Graph:"
for i in G: print i

print sum(len(c) for c in G)