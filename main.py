import numpy as np
import pandas as pd
import pyreadstat as prs
import os 
import DE_Utilities as DE


def main():
    DE.xpt_to_csv("datasets" , "datasets_csv")
    DE.join_datasets("datasets_csv" , output_dir = "bronze" , first_file="ACQ_L.csv" , join_on='SEQN')

    DF = DE.extract_csv("bronze/final_data.csv")
    final_df , X , Y = DE.transform(DF)
    DE.load(("silver/processed.csv" , "gold/X.csv" , "gold/Y.csv") , (final_df , X , Y))
    




if __name__ == "__main__":

    main()