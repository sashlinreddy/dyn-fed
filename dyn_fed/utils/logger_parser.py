"""Utility that parses logs to extract accuracies and times
"""
import glob
import os
import re

import click
import pandas as pd


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Main logic for logger parser
    """
    logd = os.path.join(input_filepath, "*/*logfile.*log")
    files = glob.glob(logd)

    configs = [
        "N_WORKERS", "COMM_PERIOD", "AGG_MODE", "COMM_MODE",
        "NONIID", "UNBALANCED"
    ]

    metrics = ["TIME", "ACCURACY", "PKT_SIZE"]

    columns = configs + metrics
    results = pd.DataFrame(
        columns=columns
    )

    for i, filename in enumerate(files):
        with open(filename, "r") as f:
            logfile = f.read()

            enc_run_name_idx = logfile.find("Starting run: ")
            if enc_run_name_idx >= 0:
                match = re.search(
                    r"(?<=Starting run: ).+?(?=\n)",
                    logfile
                )
                if match:
                    enc_run_name = match.group()
                    enc_list = enc_run_name.split("-")
                    results.loc[i, "N_WORKERS"] = enc_list[0]
                    results.loc[i, "COMM_PERIOD"] = int(enc_list[9])
                    # results.loc[i, "OVERLAP"] = enc_list[7]
                    results.loc[i, "COMM_MODE"] = enc_list[10]
                    results.loc[i, "AGG_MODE"] = enc_list[8]
                    if len(enc_list) > 11:
                        results.loc[i, "UNBALANCED"] = enc_list[12]
                        results.loc[i, "NONIID"] = enc_list[11]
                    else:
                        results.loc[i, "UNBALANCED"] = 0

            iteration_match = re.search(
                r"(?<=iterations is ).+?(?=s)", logfile
            )
            if iteration_match:
                time_sec = iteration_match.group()
                results.loc[i, "TIME"] = time_sec
            else:
                print(f'No time match for {enc_run_name}')

            accuracy_match = re.search(
                r"(?<=Accuracy=).+?(?=%)",
                logfile
            )
            if accuracy_match:
                accuracy = accuracy_match.group()
                results.loc[i, "ACCURACY"] = accuracy
            else:
                print(f'No accuracy match for {enc_run_name}')

            pkt_size_match = re.search(
                r"(?<=Total packet size communicated=).+?(?=MB)",
                logfile
            )
            if pkt_size_match:
                pkt_size = pkt_size_match.group()
                results.loc[i, "PKT_SIZE"] = pkt_size
            else:
                print(f'No pkt size match for {enc_run_name}')

            print(f"time={time_sec}, acc={accuracy}, pkt_size={pkt_size}")

    results = results.sort_values(by=configs).reset_index(drop=True)
    print(results.head(10))

    results.to_csv(output_filepath, index=False)

if __name__ == "__main__":

    main() # pylint: disable=no-value-for-parameter
