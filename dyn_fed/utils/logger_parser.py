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
    logd = os.path.join(input_filepath, "*/*/*/*/logfile.*log")
    # logd = os.path.join(input_filepath, "*/*logfile.*log")
    files = glob.glob(logd)

    configs = [
        "jobid", "dataset", "model_type", "optimizer", "n_clients",
        "agg_mode", "interval", "comm_mode",
        "noniid", "unbalanced", "learning_rate",
        "epochs", "delta_threshold", "error"
    ]

    metrics = ["time", "train_acc", "test_acc", "train_loss", "test_loss", "pkt_size"]

    columns = configs + metrics
    results = pd.DataFrame(
        columns=columns
    )

    n_errors = 0

    for i, filename in enumerate(files):
        with open(filename, "r") as f:
            logfile = f.read()

            jobid_match = re.search(
                r"(?<=SLURM_JOBID=).+?(?=\n)",
                logfile
            )
            if jobid_match:
                results.loc[i, "jobid"] = jobid_match.group()
            
            match = re.search(
                r"(?<=Beginning run for ).+?(?=\n)",
                logfile
            )
            if match:
                dirname = match.group()
                splitdir = dirname.split("/")
                results.loc[i, "dataset"] = splitdir[1]
                results.loc[i, "model_type"] = splitdir[2]
                results.loc[i, "optimizer"] = splitdir[3]
                enc_run_name = splitdir[4]
                enc_list = enc_run_name.split("-")
                results.loc[i, "n_clients"] = int(enc_list[0])
                results.loc[i, "agg_mode"] = int(enc_list[3])
                results.loc[i, "interval"] = int(enc_list[4])
                results.loc[i, "comm_mode"] = int(enc_list[5])
                results.loc[i, "noniid"] = int(enc_list[6])
                results.loc[i, "unbalanced"] = int(enc_list[7])
                results.loc[i, "learning_rate"] = float(enc_list[8])
                results.loc[i, "epochs"] = int(enc_list[9])
                if len(enc_list) > 10:
                    results.loc[i, "delta_threshold"] = enc_list[10]                    

            # Time taken
            iteration_match = re.search(
                r"(?<=iterations is ).+?(?=s)", logfile
            )
            error = False
            if iteration_match:
                time_sec = iteration_match.group()
                results.loc[i, "time"] = float(time_sec)
            else:
                error = True
                print(f'No time match for {enc_run_name}')

            # Accuracy
            train_acc_match = re.search(
                r"(?<=train acc=).+?(?=%)",
                logfile
            )
            if train_acc_match:
                train_acc = train_acc_match.group()
                results.loc[i, "train_acc"] = float(train_acc)
            else:
                error = True
                # print(f'No train accuracy match for {enc_run_name}')

            test_acc_match = re.search(
                r"(?<=test acc=).+?(?=%)",
                logfile
            )
            if test_acc_match:
                test_acc = test_acc_match.group()
                results.loc[i, "test_acc"] = float(test_acc)
            else:
                error = True
                # print(f'No train accuracy match for {enc_run_name}')

            # Loss
            train_loss_match = re.search(
                r"(?<=train_loss=).+?(?=,)",
                logfile
            )
            if train_loss_match:
                train_loss = train_loss_match.group()
                results.loc[i, "train_loss"] = float(train_loss)
            else:
                error = True
                # print(f'No train accuracy match for {enc_run_name}')

            test_loss_match = re.search(
                r"(?<=test_loss=).+?(?=,)",
                logfile
            )
            if test_loss_match:
                test_loss = test_loss_match.group()
                results.loc[i, "test_loss"] = float(test_loss)
            else:
                error = True

            # Packet size
            pkt_size_match = re.search(
                r"(?<=Total packet size communicated=).+?(?=MB)",
                logfile
            )
            if pkt_size_match:
                pkt_size = pkt_size_match.group()
                results.loc[i, "pkt_size"] = float(pkt_size)
            else:
                error = True
                print(f'No pkt size match for {dirname}')

            # if not error:
            #     print(f"time={time_sec}, test_acc={test_acc}, pkt_size={pkt_size}")
            # else:
            if error:
                n_errors += 1
                results.loc[i, "error"] = 1
                print(f"error for jobid={results.loc[i, 'jobid']}")
            else:
                results.loc[i, "error"] = 0

    results = results.sort_values(by=configs).reset_index(drop=True)

    print(f"\nNo of errors found={n_errors}\n")

    print(results.head(10))

    results.to_csv(output_filepath, index=False)

if __name__ == "__main__":

    main() # pylint: disable=no-value-for-parameter
