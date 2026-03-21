import numpy as np
import pandas as pd
import os
import pyreadstat as prs
import logging as l
base_dir = os.getcwd()
dataset_dir = os.path.join(base_dir , "dataset")

logger = l.getLogger(__name__)
file_handler = l.FileHandler("logging.log")
logger.addHandler(file_handler)
logger.setLevel(l.DEBUG)
formatter = l.Formatter("%(asctime)s-%(user)s-%(level)s-%(message)s")
file_handler.setFormatter(formatter)

def extract_csv(filename):
    full_path = os.path.join(dataset_dir , filename)
    df = pd.read_csv(full_path)

    return df 

def xpt_to_csv(input_dir , output_dir):
    if not os.path.exists(input_dir):
        logger.error(f"The input folder {input_dir} doesn't exist")
        return False
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for root , dirs , files in os.walk(input_dir):
        for file in files:
            full_input_path = os.path.join(root , file)
            full_output_path = os.path.join(output_dir , file.split(".")[0] + ".csv")
            tuble = prs.read_xport(full_input_path)
            df = tuble[0]
            df.to_csv(full_output_path)
        

