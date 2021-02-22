import functools

from time import time

import cugraph, cudf

import streamlit as st

import plotly
import plotly.express as px

def algo_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            start = time()
            widget, desc_string = func(*args, **kwargs)
            end = time()
            st.write(desc_string.format(end-start))
            print(type(widget))
            if isinstance(widget, cudf.core.dataframe.DataFrame):
                st.dataframe(widget.to_pandas())
            if isinstance(widget, plotly.graph_objs._figure.Figure):
                st.plotly_chart(widget)
        except MemoryError:
            st.error("{} failed, not enough memory on this GPU".format(func.__name__[:-7]))
    return wrapper

def getWidget(dataframe, key, widget_type="Histogram"):
    if "Histogram"==widget_type:
        return px.histogram(dataframe.to_pandas(), x=key, log_y=True)
    if "Dataframe"==widget_type:
        return dataframe
    if "Dataframe Summary"==widget_type:
        if key in ["labels", "partition"]:
            return dataframe.groupby(key).count()
        else:
            return dataframe.describe()


@algo_decorator
def pagerank_filler(G, representation_type="Histogram"):
    pr = cugraph.pagerank(G, alpha=0.85, max_iter = 500, tol = 1.0e-05).set_index("vertex")
    return getWidget(pr, "pagerank", widget_type=representation_type), "## Pagerank [{:.2f}s]"

@algo_decorator
def degree_filler(G, representation_type="Histogram"):
    degree = G.degree()
    return getWidget(degree, "degree", widget_type=representation_type), "## Degree [{:.2f}]s"

# TODO: Need to improve the algo_decorator to handle multiple fig (return a dict: name: [fig, fig2, fig3, ...)
def degrees_filler(G, representation_type="Histogram"):
    start_deg = time()
    degrees = G.degrees()
    end_deg = time()
    figin = px.histogram(degrees.to_pandas(), x="in_degree", log_y=True)
    figout = px.histogram(degrees.to_pandas(), x="out_degree", log_y=True)
    st.write("## Degrees (in/out) [{:.2f}s]".format(end_deg-start_deg))
    st.plotly_chart(figin)
    st.plotly_chart(figout)

@algo_decorator
def katz_filler(G, representation_type="Histogram"):
    katz = cugraph.katz_centrality(G)
    return getWidget(katz, "katz_centrality", widget_type=representation_type), "## Katz centrality [{:.2f}s]"

## TODO: Below are clustering/community/components analyses that could benefit a better presentation

@algo_decorator
def louvain_filler(G, representation_type="Histogram"):
    louvain, modularity_score = cugraph.louvain(G.to_undirected())
    return getWidget(louvain, "partition", widget_type=representation_type), "## Louvain [{:.2f}s],"+" modularity={:.2f}".format(modularity_score)

@algo_decorator
def wcc_filler(G, representation_type="Histogram"):
    wcc = cugraph.weakly_connected_components(G)
    return getWidget(wcc, "labels", widget_type=representation_type),  "## Weakly Connected Components [{:.2f}s]"

@algo_decorator
def scc_filler(G, representation_type="Histogram"):
    scc = cugraph.strongly_connected_components(G)
    return getWidget(scc, "labels", widget_type=representation_type),  "## Strongly Connected Components [{:.2f}s]"


dispatcher = {
                "PageRank": pagerank_filler,
                "Degree": degree_filler,
                "Degrees": degrees_filler,
                "Katz Centrality": katz_filler,
                "Louvain": louvain_filler,
                "WCC":  wcc_filler,
                "SCC":  scc_filler
                }
