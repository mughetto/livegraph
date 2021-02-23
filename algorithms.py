import functools

from time import time

import cugraph, cudf

import streamlit as st

import plotly
import plotly.express as px


# This decorator could likely take care of passing config dictionnaries to the algo, as well as the widget_type?
def algo_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            start = time()
            widget_list = func(*args, **kwargs)
            for widget, desc_string in widget_list:
                end = time()
                st.write(desc_string.format(end-start))
                if isinstance(widget, cudf.core.dataframe.DataFrame):
                    st.dataframe(widget.to_pandas())
                if isinstance(widget, plotly.graph_objs._figure.Figure):
                    st.plotly_chart(widget)
        except MemoryError:
            st.error("{} failed, not enough memory on this GPU".format(func.__name__[:-7]))
    return wrapper

def getWidget(dataframe, key, widget_type="Histogram", desc_string="Missing descriptor"):
    if "Histogram"==widget_type:
        return px.histogram(dataframe.to_pandas(), x=key, log_y=True), desc_string
    if "Dataframe"==widget_type:
        return dataframe, desc_string
    if "Dataframe Summary"==widget_type:
        if key in ["labels", "partition", "cluster", "core_number"]:
            return dataframe.groupby(key).count(), desc_string
        else:
            return dataframe.describe(), desc_string


## Link analysis

@algo_decorator
def pagerank_filler(G, representation_type="Histogram", config_dict={}):
    """
    PageRank is a link analysis algorithm and it assigns a numerical weighting to each element of a hyperlinked set of documents, such as the World Wide Web, with the purpose of "measuring" its relative importance within the set.
    The algorithm may be applied to any collection of entities with reciprocal quotations and references. 
    """
    pr = cugraph.pagerank(G, **config_dict).set_index("vertex")
    return [getWidget(pr, "pagerank", widget_type=representation_type, desc_string="## Pagerank [{:.2f}s]")]

# TODO: Need to improve the algo_decorator to handle multiple fig (return a dict: name: [fig, fig2, fig3, ...)
@algo_decorator
def hits_filler(G, representation_type="Histogram", config_dict={}):
    """
    The HITS algorithm computes two numbers for a node. Authorities estimates the node value based on the incoming links. Hubs estimates the node value based on outgoing links.
    """
    hits = cugraph.hits(G, **config_dict).set_index("vertex")
    wd1 = getWidget(hits, "hubs", widget_type=representation_type, desc_string="## HITS hubs[{:.2f}s]")
    wd2 = getWidget(hits, "authorities", widget_type=representation_type, desc_string="## HITS authorities[{:.2f}s]")
    return [wd1, wd2]

## Centrality
@algo_decorator
def bc_filler(G, representation_type="Histogram", config_dict={}):
    bc = cugraph.betweenness_centrality(G).set_index("vertex")
    return [getWidget(bc, "betweenness_centrality", widget_type=representation_type, desc_string="## Betweenness centrality [{:.2f}s]")]


## Structure
@algo_decorator
def degree_filler(G, representation_type="Histogram", config_dict={}):
    """
    Compute vertex degree, which is the total number of edges incident to a vertex (both in and out edges).
    """
    degree = G.degree()
    return [getWidget(degree, "degree", widget_type=representation_type, desc_string="## Degree [{:.2f}]s")]

# TODO: Need to improve the algo_decorator to handle multiple fig (return a dict: name: [fig, fig2, fig3, ...)
def degrees_filler(G, representation_type="Histogram", config_dict={}):
    """
    Compute vertex in-degree and out-degree.
    """
    start_deg = time()
    degrees = G.degrees()
    end_deg = time()
    figin = px.histogram(degrees.to_pandas(), x="in_degree", log_y=True)
    figout = px.histogram(degrees.to_pandas(), x="out_degree", log_y=True)
    st.write("## Degrees (in/out) [{:.2f}s]".format(end_deg-start_deg))
    st.plotly_chart(figin)
    st.plotly_chart(figout)

@algo_decorator
def katz_filler(G, representation_type="Histogram", config_dict={}):
    """
    Compute the Katz centrality for the nodes of the graph G. cuGraph does not currently support the ‘beta’ and ‘weight’ parameters as seen in the corresponding networkX call. This implementation is based on a relaxed version of Katz defined by Foster with a reduced computational complexity of O(n+m)

Foster, K.C., Muth, S.Q., Potterat, J.J. et al. Computational & Mathematical Organization Theory (2001) 7: 275. https://doi.org/10.1023/A:1013470632383
    """
    katz = cugraph.katz_centrality(G)
    return [getWidget(katz, "katz_centrality", widget_type=representation_type, desc_string="## Katz centrality [{:.2f}s]")]

## Communities
@algo_decorator
def louvain_filler(G, representation_type="Histogram", config_dict={}):
    """
    Compute the modularity optimizing partition of the input graph using the Louvain method

    It uses the Louvain method described in:

    VD Blondel, J-L Guillaume, R Lambiotte and E Lefebvre: Fast unfolding of community hierarchies in large networks, J Stat Mech P10008 (2008), http://arxiv.org/abs/0803.0476
    """
    louvain, modularity_score = cugraph.louvain(G.to_undirected())
    return [getWidget(louvain, "partition", widget_type=representation_type, desc_string="## Louvain [{:.2f}s],"+" modularity={:.2f}".format(modularity_score))]


@algo_decorator
def leiden_filler(G, representation_type="Histogram", config_dict={}):
    """
    Compute the modularity optimizing partition of the input graph using the Leiden algorithm

    It uses the Louvain method described in:

    Traag, V. A., Waltman, L., & van Eck, N. J. (2019). From Louvain to Leiden: guaranteeing well-connected communities. Scientific reports, 9(1), 5233. doi: 10.1038/s41598-019-41695-z
    """
    leiden, modularity_score = cugraph.leiden(G.to_undirected())
    return [getWidget(leiden, "partition", widget_type=representation_type, desc_string="## Leiden [{:.2f}s],"+" modularity={:.2f}".format(modularity_score))]

@algo_decorator
def ecg_filler(G, representation_type="Histogram", config_dict={}):
    parts = cugraph.ecg(G)
    return [getWidget(part, "partition", widget_type=representation_type, desc_string="## Ensemble clustering [{:.2f}s]")]

## Spectral clustering
@algo_decorator
def sbcc_filler(G, representation_type="Histogram", config_dict={"num_clusters":5}):
    #n_cluster = 5
    df = cugraph.spectralBalancedCutClustering(G.to_undirected(), **config_dict)
    return [getWidget(df, "cluster", widget_type=representation_type, desc_string="## Spectral balanced cut clustering [{:.2f}s]"+" ({} clusters)".format(config_dict["num_clusters"]))]

@algo_decorator
def smmc_filler(G, representation_type="Histogram", config_dict={}):
    n_cluster = 5
    df = cugraph.spectralModularityMaximizationClustering(G.to_undirected(), n_cluster)
    return [getWidget(df, "cluster", widget_type=representation_type, desc_string="## Spectral balanced cut clustering [{:.2f}s]"+" ({} clusters)".format(n_cluster))]

## Components
@algo_decorator
def wcc_filler(G, representation_type="Histogram", config_dict={}):
    wcc = cugraph.weakly_connected_components(G)
    return [getWidget(wcc, "labels", widget_type=representation_type, desc_string="## Weakly Connected Components [{:.2f}s]")]

@algo_decorator
def scc_filler(G, representation_type="Histogram", config_dict={}):
    scc = cugraph.strongly_connected_components(G)
    return [getWidget(scc, "labels", widget_type=representation_type, desc_string="## Strongly Connected Components [{:.2f}s]")]

## Cores
@algo_decorator
def cores_filler(G, representation_type="Histogram", config_dict={}):
    cn = cugraph.core_number(G)
    return [getWidget(cn, "core_number", widget_type=representation_type, desc_string="## Core number [{:.2f}s]")]

## Link prediction
@algo_decorator
def overlap_filler(G, representation_type="Histogram", config_dict={}):
    pairs = G.get_two_hop_neighbors()
    df = cugraph.overlap(G.to_undirected(), pairs)
    return [getWidget(df, "overlap_coeff", widget_type=representation_type, desc_string="## Jaccard Coefficient [{:.2f}s]")]


dispatcher = {
                "PageRank": pagerank_filler,
                "HITS": hits_filler,
                # "Overlap": overlap_filler,
                # "Betweenness centrality": bc_filler, # too long
                "Degree": degree_filler,
                "Degrees": degrees_filler,
                "Katz Centrality": katz_filler,
                "Louvain": louvain_filler,
                "Leiden": leiden_filler,
                "Spectral balanced cut clustering": sbcc_filler,
                #"Spectral modularity maximization clustering": smmc_filler, # Needs a weighted graph
                "Weakly Connected Components":  wcc_filler,
                "Strongly Connected Components":  scc_filler,
                "Core number": cores_filler
                #"Ensemble Clustering": ecg_filler # Needs a weighted graph
                }

global_configurations = {
                "PageRank": {"alpha": 0.85, "max_iter": 100, "tol": 1e-05},
                "HITS":{"max_iter":100, "tol":1e-05},
                # "Overlap": {},
                # "Betweenness centrality": {},
                "Degree": {},
                "Degrees": {},
                "Katz Centrality": {"max_iter":100, "tol":1e-06},
                "Louvain": {"max_iter":100, "resolution":1.},
                "Leiden": {"max_iter":100, "resolution":1.},
                "Spectral balanced cut clustering": {"num_clusters": 5, "num_eigen_vects":2, "evs_tolerance":1e-05, "evs_max_iter":100, "kmean_tolerance":1e-05, "kmean_max_iter":100},
                #"Spectral modularity maximization clustering": {},
                "Weakly Connected Components":  {},
                "Strongly Connected Components":  {},
                "Core number": {}
                #"Ensemble Clustering": ecg_filler # Needs a weighted graph
                }