import networkx as nx
import pandas as pd
from variables import *

def main():
    node_df = pd.read_csv(DATA_FOLDER + NODE_VARS)
    dyad_df = pd.read_csv(DATA_FOLDER + DYAD_VARS)
    print(dyad_df.head())
    

if __name__ == "__main__":
    main()
