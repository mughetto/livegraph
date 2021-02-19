from time import time

import cugraph

import streamlit as st

import plotly.express as px



def pagerank_filler(G):
    start_pr = time()
    pr = cugraph.pagerank(G, alpha=0.85, max_iter = 500, tol = 1.0e-05).set_index("vertex")
    end_pr = time()
    figglob = px.histogram(pr.to_pandas(), x="pagerank", log_y=True)
    st.write("## Pagerank ({:.2f}s)".format(end_pr-start_pr))
    st.plotly_chart(figglob)
def degree_filler(G):
    start_deg = time()
    degree = G.degree()
    end_deg = time()
    figglob = px.histogram(degree.to_pandas(), x="degree", log_y=True)
    st.write("## Degree ({:.2f}s)".format(end_deg-start_deg))
    st.plotly_chart(figglob)
def degrees_filler(G):
    start_deg = time()
    degrees = G.degrees()
    end_deg = time()
    figin = px.histogram(degrees.to_pandas(), x="in_degree", log_y=True)
    figout = px.histogram(degrees.to_pandas(), x="out_degree", log_y=True)
    st.write("## Degrees (in/out) ({:.2f}s)".format(end_deg-start_deg))
    st.plotly_chart(figin)
    st.plotly_chart(figout)
def katz_filler(G):
    start_katz = time()
    katz = cugraph.katz_centrality(G)
    end_katz = time()
    fig = px.histogram(katz.to_pandas(), x="katz_centrality", log_y=True)
    st.write("## Katz Centrality ({:.2f}s)".format(end_katz-start_katz))
    st.plotly_chart(fig)
def louvain_filler(G):
    start_louvain = time()
    louvain, modularity_score = cugraph.louvain(G.to_undirected())
    end_louvain = time()
    fig = px.histogram(louvain.to_pandas(), x="partition", log_y=True)
    st.write("## Louvain ({:.2f}s), modularity={:.2f}".format(end_louvain-start_louvain, modularity_score))
    st.plotly_chart(fig)
    
dispatcher = {
                "PageRank": pagerank_filler,
                "Degree": degree_filler,
                "Degrees": degrees_filler,
                "Katz Centrality": katz_filler,
                "Louvain": louvain_filler
                }
