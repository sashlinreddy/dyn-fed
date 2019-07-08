import glob
import pandas as pd
import click
import os
import re

@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):

    logd = os.path.join(input_filepath, "*/*logfile.*log")
    files = glob.glob(logd)

    results = pd.DataFrame(columns=["N_WORKERS", "COMM_PERIOD", "TIME", "ACCURACY"])

    for i, filename in enumerate(files):
        with open(filename, "r") as f:
            logfile = f.read()

            enc_run_name_idx = logfile.find("Starting run: ")
            if enc_run_name_idx >= 0:
                enc_run_name_idx += 14
                enc_run_name = re.match("(.*?)\n",logfile[enc_run_name_idx:100]).group()[:-1]
                enc_list = enc_run_name.split("-")
                results.loc[i, "N_WORKERS"] = enc_list[0]
                results.loc[i, "COMM_PERIOD"] = enc_list[4]
            time_idx = logfile.find("iterations is") + 14
            time_sec = re.match("(.*?)s",logfile[time_idx:time_idx + 15]).group()[:-1]
            results.loc[i, "TIME"] = time_sec

            acc_idx = logfile.find("Accuracy=") + 9
            accuracy = logfile[acc_idx:acc_idx + 7]
            results.loc[i, "ACCURACY"] = accuracy

            print(f"time={time_sec}, acc={accuracy}")

    print(results.head())

    results.to_csv(output_filepath, index=False)

if __name__ == "__main__":

    main()