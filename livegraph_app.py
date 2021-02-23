from time import time

import pandas as pd
import numpy as np
import cugraph, cudf

from blazingsql import BlazingContext

import GPUtil

import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

import plotly.express as px

import pykeen
from pykeen.datasets import *

from algorithms import *


# TODO: add ACE code editor to edit a dedicated analysis function
# TODO: add community counts
# TODO: add WCC counts

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
    data_te = ds.testing.triples
    data = np.vstack((data_tr, data_va, data_te))
    # Load in various dataframe
    ds_df = pd.DataFrame({'head': data[:, 0], 'rel': data[:, 1], 'tail': data[:, 2]})
    ds_gdf = cudf.from_pandas(ds_df)
    bc.create_table('bio_graph', ds_gdf)
    bc.create_table(dataset_of_choice, ds_gdf.head(1)) # DEBUG: this line is here to confirm the fact that we are caching one set of blazing context and other outputs per dataset
    gdf = bc.sql("SELECT * FROM bio_graph")
    Gglob = cugraph.DiGraph()
    Gglob.from_cudf_edgelist(gdf, source='head', destination='tail')
    return bc, Gglob, dataset_of_choice

if __name__ == "__main__":
    # TODO: ad a hidable zone explaining thate timers for algo include plotting/display
    st.set_page_config(layout="wide", page_title="Live Graph Exploration")

    all_datasets = pykeen.datasets.datasets
    print("Possible datasets: ", [d for d in all_datasets.keys()])
    whitelisted_datasets = ["openbiolink", "hetionet", "codexsmall", "codexmedium", "codexlarge",  "fb15k", "wn18rr", "nations"]

    dataset_of_choice = st.sidebar.selectbox("Select one of PyKeen datasets", whitelisted_datasets)
    #selected_subsets = [ st.sidebar.checkbox(d) for d in ["testing", "training", "validating"] ]
    
    #dataset_of_choice = "hetionet"
    with st.spinner("Loading {} to GPU...".format(dataset_of_choice)):
        start_ingestion = time()
        bc, Gglob, _ = ingest(dataset_of_choice)
        end_ingestion = time()
        print(bc.list_tables())
        st.sidebar.markdown("**"+dataset_of_choice+"**"+" ingested in {:.2f}s".format(end_ingestion-start_ingestion))

    gdf = bc.sql("SELECT * FROM bio_graph ")
    unique_rels = gdf.rel.unique().to_arrow().tolist()
    print(unique_rels)
    del gdf

    # TODO: give an option to run all ? Maybe not, better have users to think about it
    #all_algos = st.sidebar.checkbox("Select all algos")

    selected_algos = st.sidebar.multiselect('Pick your cuGraph algorithm', [algo for algo in dispatcher])
    limit_sql = ""
    where_sql = ""
    with st.sidebar.beta_expander("Fine tune subgraph"):
        # TODO: push this into its own helper
        limit_sql = " LIMIT "+str(st.number_input('Max number of edges (SQL LIMIT)', min_value=1, max_value=None, value=10, step=1))
        selected_rels = st.multiselect('Predicates to exclude for the subgraph', sorted(unique_rels))
        if selected_rels:
            rel_list_to_str = [e for e in selected_rels].__repr__()[1:-1]
            where_sql = "WHERE rel NOT IN ({})".format(rel_list_to_str)
        print(where_sql)
    go_compute = st.sidebar.checkbox("Live computation")
    config_panels = st.sidebar.checkbox("Show algorithm advanced configurations")
    
    with st.sidebar.beta_expander("Modify the edge selection manually (SQL)"):
        custom_edge_sql = st.text_area('Current subgraph SQL', value="SELECT * FROM bio_graph {} {}".format(where_sql,limit_sql) )

    ## TODO: modify selection with Python/Streamlit-ace
    ## TODO: extract subgraph with cugraph.community.subgraph_extraction.subgraph(G, vertices)

    display_type = st.sidebar.radio('Display type', [ "Histogram", "Dataframe", "Dataframe Summary"])

    # TODO: Option to weight edges (Ricci, edge betweeness)
    #st.sidebar.text_area('Modify the SQL projection (?you can use features like louvain or PR here?)', value="SELECT * FROM biograph LIMIT 10")
    #st.sidebar.file_uploader('File uploader')
    #st.sidebar.color_picker('Pick a color')

    if (not selected_algos) or (not go_compute):
        st.stop()

    ## Section where we apply user-defined SQL
    gdf = bc.sql(custom_edge_sql)
    G = cugraph.DiGraph()
    G.from_cudf_edgelist(gdf, source='head', destination='tail')
    GPUs = GPUtil.getGPUs()
    st.sidebar.markdown("**GPU VRAM Utilization (max={}MB)**".format(GPUs[0].memoryTotal))
    gpu_vmem_load = st.sidebar.progress(GPUs[0].memoryUsed/GPUs[0].memoryTotal)
    st.sidebar.markdown("**GPU Utilization (max=100%)**")
    gpu_load = st.sidebar.progress(GPUs[0].load)

    #with st.beta_expander("PageRank config"):
    #    alpha_pr = st.slider("alpha", min_value=0.1, max_value=0.95, value=0.85, step=0.05)
    n_algos_to_evaluate = len(selected_algos)*2 # 2 = global graph + subgraph scenarios
    algos_evaluated = 0
    progress_bar = st.progress(algos_evaluated/n_algos_to_evaluate)
    first_pass = True


    extra_col, header_left, header_right = st.beta_columns(3)
    with header_left:
        with st.beta_expander("{} stats:".format(dataset_of_choice)):
            st.write("**Number of nodes:** {}".format(len(Gglob.nodes())))
            st.write("**Number of edges:** {}".format(len(Gglob.edges())))
            st.write("**Number of triangles:** {}".format(cugraph.triangles(Gglob.to_undirected())))
    with header_right:
        with st.beta_expander("Subgraph stats:".format(dataset_of_choice)):
            st.write("**Number of nodes:** {}".format(len(G.nodes())))
            st.write("**Number of edges:** {}".format(len(G.edges())))
            st.write("**Number of triangles:** {}".format(cugraph.triangles(G.to_undirected())))

    for algo in selected_algos:
        cols = st.beta_columns(3)
        if config_panels:
            with cols[0]:
                st.write("Config panel for {}".format(algo))
                for config in global_configurations[algo].items():
                    parameter_name = config[0]
                    current_value = global_configurations[algo][parameter_name]
                    global_configurations[algo][parameter_name] = st.number_input(parameter_name, min_value=None, max_value=None, value=global_configurations[algo][parameter_name], key=algo)
                with st.beta_expander("More on {}".format(algo)):
                    st.info(dispatcher[algo].__doc__)
        with cols[-2]:
            if first_pass:
                st.write("# Results on {}".format(dataset_of_choice))
            dispatcher[algo](Gglob, display_type, global_configurations[algo])
            algos_evaluated +=1
            progress_bar.progress(algos_evaluated/n_algos_to_evaluate)
            gpu_vmem_load.progress(GPUs[0].memoryUsed/GPUs[0].memoryTotal)
            gpu_load.progress(GPUs[0].load)   
        with cols[-1]:
            if first_pass:
                st.write("# Results on subgraph of {}".format(dataset_of_choice))
            dispatcher[algo](G, display_type)
            algos_evaluated +=1
            progress_bar.progress(algos_evaluated/n_algos_to_evaluate)
            gpu_vmem_load.progress(GPUs[0].memoryUsed/GPUs[0].memoryTotal)
            gpu_load.progress(GPUs[0].load)   
        first_pass = False

    
    gpu_vmem_load.progress(GPUs[0].memoryUsed/GPUs[0].memoryTotal)
    gpu_load.progress(GPUs[0].load)   

    st.stop() # We don't display the subgraph yet
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