import numpy as np
import pandas as pd
import os
import pyreadstat as prs
import logging as l
import sklearn.model_selection as ms
base_dir = os.getcwd()
dataset_dir = os.path.join(base_dir , "dataset")

logger = l.getLogger(__name__)
file_handler = l.FileHandler("logging.log")
logger.addHandler(file_handler)
logger.setLevel(l.DEBUG)
formatter = l.Formatter("%(asctime)s-%(levelname)s-%(message)s")
file_handler.setFormatter(formatter)

def extract_csv(filename):
    try:
        logger.info(f"File:{filename} is being extracted...")
        full_path = filename
        df = pd.read_csv(full_path)
        logger.info(f"File:{filename} was extracted successfully")
        return df 
    except Exception as e:
        logger.error(f"The file:{filename} wasn't extracted , Error:{e}")
        return None

def xpt_to_csv(input_dir , output_dir):
    if not os.path.exists(input_dir):
        logger.error(f"The input folder {input_dir} doesn't exist")
        return False
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for root , dirs , files in os.walk(input_dir):
        for file in files:
            try:
                full_input_path = os.path.join(root , file)
                full_output_path = os.path.join(output_dir , file.split(".")[0] + ".csv")
                tuble = prs.read_xport(full_input_path , encoding = "cp1252")
                df = tuble[0]
                df.to_csv(full_output_path)
            except Exception as e:
                logger.warning(f"file:{file} was not processed due to error :{e}")

def join_datasets(input_dir , output_dir , first_file , join_on):
    if not os.path.exists(input_dir):
        logger.error(f"The input folder {input_dir} doesn't exist")
        return False
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    final_df = pd.read_csv(os.path.join(input_dir , first_file))
    i = 0
    full_output_path = os.path.join(output_dir , "final_data.csv")
    for root , dirs , files in os.walk(input_dir):

        for file in files:
        
            try:
                i += 1
                full_input_path = os.path.join(root , file)
                if file != first_file:
                    df = pd.read_csv(full_input_path)
                    cols = list(df.columns)
                    cols.remove('Unnamed: 0')
                    df = df.loc[: , cols]
                    final_df = final_df.merge(df , on = join_on ,  how = "outer")
            except Exception as e:
                logger.warning(f"file:{file} was not processed due to error :{e}")
    
    final_df.to_csv(full_output_path)


def transform(df , test_size = .2):
    try:
        logger.info("The data is being processed...")
        required_cols = ['BPQ020' , 'RIAGENDR' , "RIDAGEYR"  , "BMXBMI", "PAD790Q" , "DIQ010" , "DR1TSODI" ] 
        required_cols_mapping = ["had_hypertension" , "gender" , "age_at_years"   , "Body_Mass_Index"   , "Frequency_of_moderate_LTPA" , "had_diabetes"  , "Sodium_(mg)_perday"]
        df_f_v_1 = df.loc[: , required_cols]
        df_f_v_1.columns = required_cols_mapping
        df_f_v_2 = df_f_v_1[~df_f_v_1.had_hypertension.isna()]
        df_f_v_3 = df_f_v_2
        BMI_mean = df_f_v_3.Body_Mass_Index.mean()
        df_f_v_4 = df_f_v_3.fillna(BMI_mean)
        df_f_v_5 = df_f_v_4[(df_f_v_4.had_hypertension == 1.0) | (df_f_v_4.had_hypertension == 2.0)]
        df_f_v_5.had_hypertension = df_f_v_5.had_hypertension.apply(lambda row : 1 if row == 1.0 else 0)
        df_f_v_5.gender = df_f_v_5.gender.apply(lambda row : 1 if row == 1.0 else 0)
        df_f_v_6 = df_f_v_5.loc[(df_f_v_5.had_diabetes == 1.0)|(df_f_v_5.had_diabetes == 2.0)]
        df_f_v_6.had_diabetes = df_f_v_6.had_diabetes.apply(lambda row : 1 if row == 1.0 else 0)
        df_f_v_7 = df_f_v_6.rename(columns = {"had_hypertension" : "target"})
        min_target = df_f_v_7.groupby('target').count().gender.min()
        class_1 = df_f_v_7.loc[df_f_v_7.target == 1].iloc[:min_target]
        class_0 = df_f_v_7.loc[df_f_v_7.target == 0].iloc[:min_target]
        df_f_v_final = pd.concat([class_1 , class_0]).astype("float")
        

        cols = df_f_v_final.columns[1:]

        X = df_f_v_final.iloc[: , 1:]
        Y = df_f_v_final.iloc[: , 0]
        X_train , X_test , Y_train , Y_test = ms.train_test_split(X , Y , test_size = test_size, random_state=42 )


        for col in cols:
            mean_value = X_train.loc[: , col].mean()
            max_value = X_train.loc[: , col].max()
            min_value = X_train.loc[: , col].min()
            result_train = (X_train.loc[: , col] - np.array(mean_value))/(np.array(max_value) - np.array(min_value))
            result_test  = (X_test.loc[: , col] - np.array(mean_value))/(np.array(max_value) - np.array(min_value))
            X_train.loc[: , col] = result_train
            X_test.loc[: , col] = result_test

        logger.info("The has been processed successfully")
        return df_f_v_final , X , Y , X_train.to_numpy() , X_test.to_numpy() , Y_train.to_numpy() , Y_test.to_numpy()
    except Exception as e:
        logger.error("The data hasn't been processed successfully")
        return None , None , None
    

def load(destinations , data):
    
    
    if len(destinations) != len(data):
        logger.error("The length of destinations is not equal to length of data")
        return None
    logger.info(f"The data is being loaded to destinations")
    try:
        i = 0 
        while i < len(destinations):
            df = data[i]
            dest = destinations[i]
            dest_list = dest.split("/")
            if len(dest_list) != 1:
                dirs = dest_list[:-1]
                directory = "/".join(dirs)
                if not os.path.exists(directory):
                    os.makedirs(directory)
            df.to_csv(dest)
            i +=1
        logger.info(f"The data has been loaded to destinations successfully")
    except Exception as e:
        logger.error(f"The data hasn't been loaded successfully , Error : {e}")
    return None