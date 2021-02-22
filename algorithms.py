from time import time

import cugraph

import streamlit as st

import plotly.express as px

def algo_decorator(func):
    def wrapper(*args, **kwargs):
        try:
            start = time()
            fig, desc_string = func(*args, **kwargs)
            end = time()
            st.write(desc_string.format(end-start))
            st.plotly_chart(fig)
        except MemoryError:
            st.error("{} failed, not enough memory on this GPU".format(func.__name__[:-7]))
    return wrapper

@algo_decorator
def pagerank_filler(G):
    pr = cugraph.pagerank(G, alpha=0.85, max_iter = 500, tol = 1.0e-05).set_index("vertex")
    fig = px.histogram(pr.to_pandas(), x="pagerank", log_y=True)
    return fig, "## Pagerank [{:.2f}s]"

@algo_decorator
def degree_filler(G):
    degree = G.degree()
    fig = px.histogram(degree.to_pandas(), x="degree", log_y=True)
    return fig, "## Degree [{:.2f}]s"

# TODO: Need to improve the algo_decorator to handle multiple fig (return a dict: name: [fig, fig2, fig3, ...)
def degrees_filler(G):
    start_deg = time()
    degrees = G.degrees()
    end_deg = time()
    figin = px.histogram(degrees.to_pandas(), x="in_degree", log_y=True)
    figout = px.histogram(degrees.to_pandas(), x="out_degree", log_y=True)
    st.write("## Degrees (in/out) [{:.2f}s]".format(end_deg-start_deg))
    st.plotly_chart(figin)
    st.plotly_chart(figout)

@algo_decorator
def katz_filler(G):
    katz = cugraph.katz_centrality(G)
    fig = px.histogram(katz.to_pandas(), x="katz_centrality", log_y=True)
    return fig, "## Katz centrality [{:.2f}s]"

## TODO: Below are clustering/community/components analyses that could benefit a better presentation

@algo_decorator
def louvain_filler(G):
    louvain, modularity_score = cugraph.louvain(G.to_undirected())
    fig = px.histogram(louvain.to_pandas(), x="partition", log_y=True)
    return fig, "## Louvain [{:.2f}s],"+" modularity={:.2f}".format(modularity_score)

@algo_decorator
def wcc_filler(G):
    wcc = cugraph.weakly_connected_components(G)
    fig = px.histogram(wcc.to_pandas(), x="labels", log_y=True)
    return fig, "## Weakly Connected Components [{:.2f}s]"

@algo_decorator
def scc_filler(G):
    scc = cugraph.strongly_connected_components(G)
    fig = px.histogram(scc.to_pandas(), x="labels", log_y=True)
    return fig, "## Strongly Connected Components [{:.2f}s]"


dispatcher = {
                "PageRank": pagerank_filler,
                "Degree": degree_filler,
                "Degrees": degrees_filler,
                "Katz Centrality": katz_filler,
                "Louvain": louvain_filler,
                "WCC":  wcc_filler,
                "SCC":  scc_filler
                }
