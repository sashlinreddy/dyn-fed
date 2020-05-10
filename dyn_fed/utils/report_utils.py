"""Utility functions for generating analysis on results
"""
import re
from typing import List, Collection
from collections import defaultdict

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
    dyn_avg1 = r'DynAvg $\Delta=0.3$' if is_sgd else r'DynAvg $\Delta=4.3$'
    dyn_avg2 = r'DynAvg $\Delta=0.8$' if is_sgd else r'DynAvg $\Delta=4.8$'

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
    dyn_avg1 = r'DynAvg $\Delta=0.3$' if is_sgd else r'DynAvg $\Delta=4.3$'
    dyn_avg2 = r'DynAvg $\Delta=0.8$' if is_sgd else r'DynAvg $\Delta=4.8$'

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

def parse_logs(files, lookup_date):
    idx = files[0].split("/").index(lookup_date)
    mydict = lambda: defaultdict(mydict)
    results = mydict()
    count = 0
    for fname in files:
        split_path = fname.split("/")
        client_match = re.search(
            r"(?<=log-).+",
            split_path[-1].split(".")[0]
        )
        if client_match:
            split_path[-1] = client_match.group()
        name = "/".join(split_path[idx:])
        with open(fname, "r") as f:
            logfile = f.read()
            train_losses = re.findall(
                r"(?<=train_loss= ).+?(?=,)",
                logfile
            )

            test_losses = re.findall(
                r"(?<=test_loss= ).+?(?=,)",
                logfile
            )
            train_losses = list(map(lambda x: float(x), train_losses))
            test_losses = list(map(lambda x: float(x), test_losses))
            results[name]["train"] = train_losses
            results[name]["test"] = test_losses  
            count += 1

    print(f"Parsed {count} files")

    return results

def extract_values(obj, query):
    """Pull all values of specified key from nested JSON."""
    result = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            condition = True
            split = k.split("/")
            date = split[-6]
            dataset = split[-5]
            model_type = split[-4]
            optimizer = split[-3]
            run_name = split[-2]
            split_run = run_name.split("-")
            if len(split_run) == 10:
                (n_workers,
                 scenario,
                 quantize,
                 agg_mode,
                 interval,
                 mode,
                 noniid,
                 unbalanced,
                 learning_rate,
                 n_iterations
                ) = split_run
            else:
                (n_workers,
                 scenario,
                 quantize,
                 agg_mode,
                 interval,
                 mode,
                 noniid,
                 unbalanced,
                 learning_rate,
                 n_iterations,
                 delta
                ) = split_run

            condition = eval(query)
            if condition:
                result.append({k: v})
    return result
