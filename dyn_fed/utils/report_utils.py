"""Utility functions for generating analysis on results
"""
from typing import List, Collection

import numpy as np
import pandas as pd

ListOfPandas = Collection[pd.DataFrame]
ListOfPandasList = Collection[ListOfPandas]

def performance_pivot(df, query, value, columns=None, xlabel=None):
    """Generates performance pivot for a certain value. Eg. loss or accuracy
    """
    avg_piv = (
        df.query(query + 'and interval > 1')
        .pivot_table(
            values=value,
            index="n_clients",
            columns=['comm_mode', 'interval']
        )
    )
    piv = (
        df.query(query + 'and interval < 2')
        .pivot_table(
            values=value,
            index="n_clients",
            columns=['comm_mode', 'delta_threshold']
        )
    )
    merged = pd.concat([piv, avg_piv], axis=1)
    
    if columns is not None:
        merged.columns = columns
    if xlabel is not None:
        merged.index.name = xlabel
    
    return merged

def generate_all_pivots(df: pd.DataFrame, metrics: List) -> ListOfPandasList:
    """Generate all performance pivots for iid and balanced; iid and unbalanced;
    noniid and balanced; noniid and unbalanced;
    """
    xlabel = 'No. of clients'
    is_sgd = (df["optimizer"] == "sgd").all()
    dyn_avg1 = r'DynAvg $\Delta=0.3$' if is_sgd else r'DynAvg $\Delta=2.2$'
    dyn_avg2 = r'DynAvg $\Delta=0.8$' if is_sgd else r'DynAvg $\Delta=2.8$'

    columns = [
        r'FedAvg, $\rho=1$', r'DynAvg SVD', r'DynAvg Loss', dyn_avg1, dyn_avg2,
        r'FedAvg, $\rho=10$', r'FedAvg, $\rho=20$', r'FedAvg, $\rho=50$',
        r'FedAvg, $\rho=100$'
    ]
    if df["model_type"].isin(["nn1", "cnn1"]).all():
        columns = [
            r'FedAvg, $\rho=1$', r'DynAvg SVD', r'DynAvg Loss', dyn_avg1,
            dyn_avg2, r'FedAvg, $\rho=10$', r'FedAvg, $\rho=50$'
        ]
    
    results = []
    queries = []
    for i in np.arange(2):
        for j in np.arange(2):
            query = f'noniid == {i} and unbalanced == {j}'
            queries.append(query)
            tmp = {}
            for m in metrics:
                p = performance_pivot(df, query, m, columns=columns, xlabel=xlabel)
                tmp[m] = p
            results.append(tmp)

    return results, queries

def generate_packet_size_pivots(df: pd.DataFrame):
    """Generate packet size pivots
    """
    pkt_size_piv = (
        df
        .pivot_table(
            values="pkt_size", index="n_clients", columns=["comm_mode", "interval", "delta_threshold"]
        )
    )
    is_sgd = (df["optimizer"] == "sgd").all()
    dyn_avg1 = r'DynAvg $\Delta=0.3$' if is_sgd else r'DynAvg $\Delta=2.2$'
    dyn_avg2 = r'DynAvg $\Delta=0.8$' if is_sgd else r'DynAvg $\Delta=2.8$'

    columns = [
        r'FedAvg, $\rho=1$', r'FedAvg, $\rho=10$', r'FedAvg, $\rho=20$',
        r'FedAvg, $\rho=50$', r'FedAvg, $\rho=100$', r'DynAvg SVD',
        r'DynAvg Loss', dyn_avg1, dyn_avg2
    ]
    if df["model_type"].isin(["nn1", "cnn1"]).all():
        columns = [
            r'FedAvg, $\rho=1$', r'FedAvg, $\rho=10$', r'FedAvg, $\rho=50$',
            r'DynAvg SVD', r'DynAvg Loss', dyn_avg1, dyn_avg2
        ]

    pkt_size_piv.columns = columns
    if df["model_type"].isin(["nn1", "cnn1"]).all():
        cols = [
            r'FedAvg, $\rho=1$', r'DynAvg SVD',
            r'DynAvg Loss', dyn_avg1, dyn_avg2,
            r'FedAvg, $\rho=10$', r'FedAvg, $\rho=20$',
            r'FedAvg, $\rho=50$', r'FedAvg, $\rho=100$'
        ]
        pkt_size_piv = pkt_size_piv.loc[:, cols]
    else:
        cols = [
            r'FedAvg, $\rho=1$', r'DynAvg SVD',
            r'DynAvg Loss', dyn_avg1, dyn_avg2,
            r'FedAvg, $\rho=10$', r'FedAvg, $\rho=20$',
            r'FedAvg, $\rho=50$', r'FedAvg, $\rho=100$'
    ]
        pkt_size_piv = pkt_size_piv.loc[:, cols]
    pkt_size_piv.index = pkt_size_piv.index - 1
    return pkt_size_piv
