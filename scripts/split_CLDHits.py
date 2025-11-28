"""Splits the CLDHits dataset such that each file contains a single event.
Call with 'python3'

Usage: split_CLDHits.py [-i STR | --input_dir=STR] [-o STR | --output_dir=STR]

Options:
  -i <input_dir>, --input_dir=<input_dir>
      Path to the CLDHits dataset (multiple events per file).
      [default: /home/laurits/MLPF/particlemind/data/p8_ee_tt_ecm365/parquet/]

  -o <output_dir>, --output_dir=<output_dir>
      Path to the directory where the one-event-per-file dataset is written.
      [default: CLDHits_OEPF]


"""

import os
import glob
import docopt
import awkward as ak
from tqdm import tqdm


def split(input_dir: str, output_dir: str):
    """Splits the CLDHits dataset files into multiple files such that each file
    contains a single event

    Parameters:
        input_dir : str
            Directory of the CLDHits dataset
        output_dir : str
            Directory where the one-event-per-file dataset is written

    Returns:
        None

    """
    input_files_wcp = glob.glob(os.path.join(input_dir, "*"))
    os.makedirs(output_dir, exist_ok=True)
    print("Splitting files: ")
    with tqdm(input_files_wcp) as pbar:
        for path in pbar:
            pbar.set_postfix_str(f"Processing: {path}")
            data = ak.from_parquet(path)
            filename = os.path.basename(path).split(".")[0]
            data = ak.Array({key: data[key] for key in data.fields})
            new_dir = os.path.join(output_dir, filename)
            os.makedirs(new_dir, exist_ok=True)
            for idx in range(len(data["genparticle_to_calo_hit_matrix"])):
                new_file_path = os.path.join(new_dir, f"{idx}.parquet")
                event = data[idx]
                ak.to_parquet(event, new_file_path)


if __name__ == "__main__":
    try:
        arguments = docopt.docopt(__doc__)
        input_dir_ = arguments.get("--input_dir")
        output_dir_ = arguments.get("--output_dir")
        split(input_dir_, output_dir_)
    except docopt.DocoptExit as e:
        print(e)
