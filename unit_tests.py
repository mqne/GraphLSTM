import rnn_cell_impl as rci
import networkx as nx
#import graph as rci_graph

# test graph: 20 nodes
_kickoff_hand = [("t0", "wrist"), ("i0", "wrist"), ("m0", "wrist"), ("r0", "wrist"), ("p0", "wrist"), ("i0", "m0"),
                ("m0", "r0"), ("r0", "p0"), ("t0", "t1"), ("t1", "t2"), ("i0", "i1"), ("i1", "i2"), ("i2", "i3"),
                ("m0", "m1"), ("m1", "m2"), ("m2", "m3"), ("r0", "r1"), ("r1", "r2"), ("r2", "r3"), ("p0", "p1"),
                ("p1", "p2"), ("p2", "p3")]

def test_init_GraphLSTMNet():
    G = nx.Graph(_kickoff_hand)

    try: rci.GraphLSTMNet(3)
    except TypeError: pass
    else: print "01 GraphLSTMNet did not raise TypeError when init'd with something else than a nx.Graph"

    try: rci.GraphLSTMNet(None)
    except ValueError: pass
    else: print "02 GraphLSTMNet did not raise ValueError when init'd with None as graph"

    try: rci.GraphLSTMNet()
    except TypeError: pass
    else: print "03 GraphLSTMNet did not raise TypeError when constructed with empty constructor"

    gnet = rci.GraphLSTMNet(G)

    try: gnet._cell("_")
    except KeyError: pass
    else: print "05 GraphLSTMNet._cell() did not raise KeyError for non-existing node"

    try: gnet._cell("wrist")
    except KeyError: pass
    else: print "04 GraphLSTMNet._cell() did not raise KeyError for node without cell"

    print gnet._graph["wrist"]['cell']

    a = gnet._cell("wrist")
    if a is not 123: print "05 GraphLSTMNet._cell() did not return expected value, but: %s" % str(a)




def main():
    test_init_GraphLSTMNet()
    print "All tests done."


main()
