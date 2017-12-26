import rnn_cell_impl as rci
import networkx as nx
import graph as rci_graph

# test graph: 20 nodes
_kickoff_hand = [("t0", "wrist"), ("i0", "wrist"), ("m0", "wrist"), ("r0", "wrist"), ("p0", "wrist"), ("i0", "m0"),
                ("m0", "r0"), ("r0", "p0"), ("t0", "t1"), ("t1", "t2"), ("i0", "i1"), ("i1", "i2"), ("i2", "i3"),
                ("m0", "m1"), ("m1", "m2"), ("m2", "m3"), ("r0", "r1"), ("r1", "r2"), ("r2", "r3"), ("p0", "p1"),
                ("p1", "p2"), ("p2", "p3")]


def test_init_GraphLSTMNet():
    G = nx.Graph(_kickoff_hand)

    try: rci.GraphLSTMNet(3)
    except TypeError: pass
    else: print "00 GraphLSTMNet did not raise TypeError when init'd with something else than a nx.Graph"

    try: rci.GraphLSTMNet(None)
    except ValueError: pass
    else: print "01 GraphLSTMNet did not raise ValueError when init'd with None as graph"

    try: rci.GraphLSTMNet()
    except TypeError: pass
    else: print "02 GraphLSTMNet did not raise TypeError when constructed with empty constructor"

    gnet = rci.GraphLSTMNet(G)

    try: gnet._cell("_")
    except KeyError: pass
    else: print "03 GraphLSTMNet._cell() did not raise KeyError for non-existing node"

    try: gnet._cell("wrist")
    except KeyError: pass
    else: print "04 GraphLSTMNet._cell() did not raise KeyError for node without cell"

    return gnet

def test__cell_GraphLSTMNet(gnet=None):
    if gnet is None:
        gnet = rci.GraphLSTMNet(nx.Graph(_kickoff_hand))

    gnet._graph.node["wrist"]["cell"] = 123
    a = gnet._cell("wrist")

    if a is not 123: print "10 GraphLSTMNet._cell() did not return expected value (int:123), but: %s" % str(a)

    glcell = rci.GraphLSTMCell
    b = glcell(1)
    gnet._graph.node["t0"]["cell"] = b
    a = gnet._cell("t0")

    if a is not b: print "11 GraphLSTMNet._cell() did not return expected value (%s), but: %s" % (str(b), str(a))

    b = glcell(1)

    if a is b: print "12 GraphLSTMNet._cell() returned cell (%s) that could not be told apart " \
                     "from freshly generated one (%s)" % (str(a), str(b))

    return gnet


# print node information for graph or GraphLSTMNet G
def print_node(name, G):
    if isinstance(G, rci.GraphLSTMNet):
        g = G._graph
        print "Node information for GraphLSTMNet %s:" % str(G)
    else:
        g = G
        print "Node information for graph %s:" % str(G)
    print "G[\"%s\"]: %s" % (name, str(g["wrist"]))
    print "G.node[\"%s\"]: %s" % (name, str(g.node["wrist"]))


def main():
    # rci_graph.main()
    gnet = test_init_GraphLSTMNet()
    test__cell_GraphLSTMNet(gnet)
    print "All tests done."


main()
