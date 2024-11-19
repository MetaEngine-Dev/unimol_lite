import pandas as pd
from tqdm import tqdm
import lmdb
import pickle
from multiprocessing import Pool, cpu_count
from unimol_lite.data.prepare_unimol_item import smiles2unimolitem

def generate_item(smiles):
    try:
        item = {
            "smiles": smiles,
            "item": smiles2unimolitem(smiles),
        }
    except Exception as e:
        item = {
            "smiles": smiles,
            "item": {},
            "error": str(e),
        }
    return item

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input CSV")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to save the LMDB")
    parser.add_argument("-n", "--nproc", type=int, default=cpu_count(), help="Number of processes to use")
    parser.add_argument("-s", "--sep", type=int, default=1000, help="Number of items to process in parallel")
    parser.add_argument("-c", "--col", type=str, default="CANON_SMILES", help="Column name of SMILES")
    args = parser.parse_args()
    hmdb = pd.read_csv(args.input).drop_duplicates(args.col)
    env = lmdb.open(
        args.output, 
        map_size=1099511627776,
        subdir=False,
        readonly=False,
        meminit=False,
    )
    with env.begin(write=True) as txn, Pool(cpu_count()) as p:
        sep = args.sep
        item_count = 0
        for i in tqdm(range(0, len(hmdb), sep)):
            items = list(tqdm(p.imap(generate_item, hmdb[args.col][i:i+sep]), total=sep))
            for item in items:
                txn.put(str(item_count).encode(), pickle.dumps(item))
                item_count += 1
    env.close()