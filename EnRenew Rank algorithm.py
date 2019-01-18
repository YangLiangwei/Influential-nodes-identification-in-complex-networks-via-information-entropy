
# coding: utf-8

# In[1]:

def EnRenewRank(G, topk, order):
    # N - 1
    all_degree = nx.number_of_nodes(G) - 1
    # avg degree
    k_ = nx.number_of_edges(G) * 2 / nx.number_of_nodes(G)
    # E<k>
    k_entropy = - k_ * ((k_ / all_degree) * math.log((k_ / all_degree)))

    # node's information pi
    node_information = {}
    for node in nx.nodes(G):
        information = (G.degree(node) / all_degree)
        node_information[node] = - information * math.log(information)

    # node's entropy Ei
    node_entropy = {}
    for node in nx.nodes(G):
        node_entropy[node] = 0
        for nbr in nx.neighbors(G, node):
            node_entropy[node] += node_information[nbr]

    rank = []
    for i in range(topk):
        # choose the max entropy node
        max_entropy_node, entropy = max(node_entropy.items(), key=lambda x: x[1])
        rank.append((max_entropy_node, entropy))

        cur_nbrs = nx.neighbors(G, max_entropy_node)
        for o in range(order):
            for nbr in cur_nbrs:
                if nbr in node_entropy:
                        node_entropy[nbr] -= (node_information[max_entropy_node] / k_entropy) / (2**o)
            next_nbrs = []
            for node in cur_nbrs:
                nbrs = nx.neighbors(G, node)
                next_nbrs.extend(nbrs)
            cur_nbrs = next_nbrs

        #set the information quantity of selected nodes to 0
        node_information[max_entropy_node] = 0
        # set entropy to 0
        node_entropy.pop(max_entropy_node)
    return rank

