from time import time

import _thread

import pandas as pd
import numpy as np
import cugraph, cudf

from blazingsql import BlazingContext

import streamlit as st
import streamlit_theme as stt
from streamlit_agraph import agraph, Node, Edge, Config

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import plotly.express as px

import pykeen
from pykeen.datasets import *

from algorithms import *


# TODO: we cache the complete blazing context! Need to evolve toward a singleton class for future perf? how does this caching work with GPU  mem ? any risk of useless blowup?
# Skipping the hash of dict avoids a downstream issues with _thread.lock hashing
# COOLSTUFF: streamlit cache is shared across users
@st.cache(hash_funcs={str: hash, dict: lambda _: None },allow_output_mutation=True)
def ingest(dataset_of_choice: str):
    #ds = Hetionet()#CoDExSmall()
    bc = BlazingContext()
    klass = all_datasets[dataset_of_choice]#pykeen.datasets.hetionet.Hetionet()
    ds = klass()
    data_tr = ds.training.triples
    data_va = ds.validation.triples
    data = np.vstack((data_tr, data_va))
    # Load in various dataframe
    ds_df = pd.DataFrame({'head': data[:, 0], 'rel': data[:, 1], 'tail': data[:, 2]})
    ds_gdf = cudf.from_pandas(ds_df)
    bc.create_table('biograph', ds_gdf)
    bc.create_table(dataset_of_choice, ds_gdf.head(1)) # DEBUG: this line is here to confirm the fact that we are caching one set of blazing context and other outputs per dataset
    gdf = bc.sql("SELECT * FROM biograph")
    Gglob = cugraph.DiGraph()
    Gglob.from_cudf_edgelist(gdf, source='head', destination='tail')
    return bc, Gglob, dataset_of_choice

if __name__ == "__main__":
    # TODO: ad a hidable zone explaining thate timers for algo include plotting/display
    st.set_page_config(layout="wide", page_title="Live Graph Exploration")
    stt.set_theme({'primary': '#003388'})

    all_datasets = pykeen.datasets.datasets
    print("Possible datasets: ", [d for d in all_datasets.keys()])
    whitelisted_datasets = ["codexsmall", "codexmedium", "codexlarge", "hetionet", "fb15k", "wn18rr"]

    dataset_of_choice = st.sidebar.selectbox("Select one of PyKeen datasets", sorted(whitelisted_datasets))
    #dataset_of_choice = "hetionet"
    with st.spinner("Loading {} to GPU...".format(dataset_of_choice)):
        start_ingestion = time()
        bc, Gglob, _ = ingest(dataset_of_choice)
        end_ingestion = time()
        print(bc.list_tables())
        st.sidebar.markdown("**"+dataset_of_choice+"**"+" ingested in {:.2f}s".format(end_ingestion-start_ingestion))

    st.sidebar.write("Number of nodes: {}".format(len(Gglob.nodes())))
    st.sidebar.write("Number of edges: {}".format(len(Gglob.edges())))
    selected_algos = st.sidebar.multiselect('Multiselect', ["Degree", "Degrees", "PageRank",  "Katz Centrality", "Louvain"])
    go_compute = st.sidebar.checkbox("Allow computation")
    st.sidebar.radio('Display type', [ "Histogram", "Dataframe"])
    custom_edge_sql = st.sidebar.text_area('Modify the SQL projection', value="SELECT * FROM biograph LIMIT 10" )
    #st.sidebar.number_input('SQL LIMIT', min_value=1, max_value=100, value=10, step=1)
    #st.sidebar.text_area('Modify the SQL projection (?you can use features like louvain or PR here?)', value="SELECT * FROM biograph LIMIT 10")
    #st.sidebar.file_uploader('File uploader')
    #st.sidebar.color_picker('Pick a color')

    if (not selected_algos) or (not go_compute):
        st.stop()

    ## Section where we apply user-defined SQL
    gdf = bc.sql(custom_edge_sql)
    #print(gdf[1:], type(gdf), gdf.columns)
    #st.dataframe(gdf.to_pandas())
    G = cugraph.DiGraph()
    G.from_cudf_edgelist(gdf, source='head', destination='tail')
    #st.dataframe(cugraph.weakly_connected_components(G).to_pandas())#.groupby("labels").count())

    #with st.beta_expander("PageRank config"):
    #    alpha_pr = st.slider("alpha", min_value=0.1, max_value=0.95, value=0.85, step=0.05)
    n_algos_to_evaluate = len(selected_algos)*2
    algos_evaluated = 0
    progress_bar = st.progress(algos_evaluated/n_algos_to_evaluate)
    first_pass = True

    for algo in selected_algos:
        cols = st.beta_columns(2)
        with cols[0]:
            if first_pass:
                st.write("# Complete {}".format(dataset_of_choice))
            dispatcher[algo](Gglob)
            algos_evaluated +=1
            progress_bar.progress(algos_evaluated/n_algos_to_evaluate)
        with cols[1]:
            if first_pass:
                st.write("# Info on the selected subgraph")
            dispatcher[algo](G)
            algos_evaluated +=1
            progress_bar.progress(algos_evaluated/n_algos_to_evaluate)
        first_pass = False
    st.stop()
    # Link graph size to pagerank: size=pr.to_pandas().loc[el[1]].iloc[0]*3000
    # TODO: get icons locally, adapt them to each type wehn possible to infer from node name
    # svg="https://www.exploregenetherapy.com/images/icon/dna-helix.svg"
    nodes = [ Node(id=el[1], label=el[1], size=200) for el in G.nodes().to_pandas().items()]
    edges = [ Edge(source=el[1][0], label=el[1][1], target=el[1][2], type="CURVE_SMOOTH")  for el in gdf.to_pandas().iterrows() ]

    config = Config(width=500, 
                    height=500, 
                    directed=True,
                    nodeHighlightBehavior=True,
                    highlightColor="#F7A7A6", # or "blue"
                    collapsible=True,
                    node={'labelProperty':'label'},
                    link={'labelProperty': 'label', 'renderLabel': True}
                    # **kwargs e.g. node_size=1000 or node_color="blue"
                    ) 
    #with col3:
    #    st.write("# Visualizing the subgraph")
    #    return_value = agraph(nodes=nodes, 
    #                  edges=edges, 
    #                  config=config)

    #st.sidebar.balloons()