import rnn_cell_impl as rci
import networkx as nx
import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl as orig_rci
#TODO what is tensorflow.python.user_ops for?
import graph as rci_graph

# test graph: 20 nodes
_kickoff_hand = [("t0", "wrist"), ("i0", "wrist"), ("m0", "wrist"), ("r0", "wrist"), ("p0", "wrist"), ("i0", "m0"),
                ("m0", "r0"), ("r0", "p0"), ("t0", "t1"), ("t1", "t2"), ("i0", "i1"), ("i1", "i2"), ("i2", "i3"),
                ("m0", "m1"), ("m1", "m2"), ("m2", "m3"), ("r0", "r1"), ("r1", "r2"), ("r2", "r3"), ("p0", "p1"),
                ("p1", "p2"), ("p2", "p3")]


_CELL = rci._CELL
_INDEX = rci._INDEX
_CONFIDENCE = rci._CONFIDENCE


# cell that always returns fixed value on call()
class DummyFixedCell(orig_rci.RNNCell):

    def __init__(self, returnValue=None, state_is_tuple=True):
        super(DummyFixedCell, self).__init__()
        self._returnValue = returnValue
        self._state_is_tuple = state_is_tuple

    @property
    def state_size(self):
        return None

    @property
    def output_size(self):
        return None

    def call(self, inputs, state, neighbour_states):
        return self._returnValue


# cell that always returns inputs, state, and neighbour_states on call()
class DummyReturnCell(orig_rci.RNNCell):

    def __init__(self, state_is_tuple=True):
        super(DummyReturnCell, self).__init__()
        self._state_is_tuple = state_is_tuple

    @property
    def state_size(self):
        return None

    @property
    def output_size(self):
        return None

    def call(self, inputs, state, neighbour_states):
        return (inputs, state, neighbour_states), (neighbour_states, state, inputs)


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

    if a != 123: print "10 GraphLSTMNet._cell() did not return expected value (int:123), but: %s" % str(a)

    glcell = rci.GraphLSTMCell
    b = glcell(1)
    gnet._graph.node["t0"]["cell"] = b
    a = gnet._cell("t0")

    if a is not b: print "11 GraphLSTMNet._cell() did not return expected object (%s), but: %s" % (str(b), str(a))

    b = glcell(1)

    if a is b: print "12 GraphLSTMNet._cell() returned cell (%s) that could not be told apart " \
                     "from freshly generated one (%s)" % (str(a), str(b))

    return gnet


def test_call_uninodal_GraphLSTMNet_notf():
    uninodal_graph = nx.Graph()
    cname = "node0"
    uninodal_graph.add_node(cname)
    gnet = rci.GraphLSTMNet(uninodal_graph)
    cell_input, cell_state_m, cell_state_h, cell_cur_output, cell_new_state = objects(5)
    gnet._graph.node[cname][_CONFIDENCE] = 0
    gnet._graph.node[cname][_INDEX] = 0

    # test correct returning of cell return value
    gnet._graph.node[cname][_CELL] = DummyFixedCell((cell_cur_output, cell_new_state)).call
    net_output = gnet.call(([cell_input]), ((cell_state_m, cell_state_h),))
    expected = ((cell_cur_output,), (cell_new_state,))
    if net_output != expected:
        print "20 GraphLSTNet.call() did not return expected objects %s, but %s. " \
              "There is probably an error in GraphLSTMNet AFTER calling the cell." % (str(expected), str(net_output))

    # test correct delivering of parameters to cell
    gnet._graph.node[cname][_CELL] = DummyReturnCell().call
    net_output = gnet.call(([cell_input]), ((cell_state_m, cell_state_h),))
    expected = (((cell_input, (cell_state_m, cell_state_h), tuple()),),
                ((tuple(), (cell_state_m, cell_state_h), cell_input),))
    if net_output != expected:
        print "21 GraphLSTNet.call() did not deliver expected objects %s to cell, but %s " \
              "If Error 20 did not appear, there is probably an error in  GraphLSTMNet BEFORE calling the cell."\
              % (str(expected), str(net_output))

    # check proper index handling
    gnet._graph.node[cname][_INDEX] = 1
    try: gnet.call(([cell_input]), ((cell_state_m, cell_state_h),))
    except IndexError: pass
    else: print "22 GraphLSTMNet with one node did not complain about index 1"


def test_call_uninodal_GraphLSTMNet_tf():
    # TODO
    #sess = tf.InteractiveSession()
    #tf.initialize_all_variables()
    raise NotImplementedError

# print node information for graph or GraphLSTMNet G
def print_node(name, G):
    if isinstance(G, rci.GraphLSTMNet):
        g = G._graph
        print "Node information for GraphLSTMNet %s:" % str(G)
    else:
        g = G
        print "Node information for graph %s:" % str(G)
    print "G[\"%s\"]: %s" % (name, str(g[name]))
    print "G.node[\"%s\"]: %s" % (name, str(g.node[name]))

# return tuple of n objects
def objects(n):
    r = []
    for _ in xrange(n):
        r.append(object())
    return tuple(r)


def main():
    test_init_GraphLSTMNet()
    test__cell_GraphLSTMNet()
    test_call_uninodal_GraphLSTMNet_notf()
    #test_call_uninodal_GraphLSTMNet_tf()
    print "All tests done."


main()
