import numpy as np
import pandas as pd
import pyreadstat as prs
import os 
import DE_Utilities as DE
import sklearn.metrics as m
import sklearn.model_selection as ms
import sklearn.linear_model as linear
from ML import Logistic_regression_model , plot_costs , plot_accuracies , save_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def main():
    DE.xpt_to_csv("datasets" , "datasets_csv")
    DE.join_datasets("datasets_csv" , output_dir = "bronze" , first_file="ACQ_L.csv" , join_on='SEQN')

    DF = DE.extract_csv("bronze/final_data.csv")
    final_df , X , Y = DE.transform(DF)
    DE.load(("silver/processed.csv" , "gold/X.csv" , "gold/Y.csv") , (final_df , X , Y))
    
    X_train , X_test , Y_train , Y_test = ms.train_test_split(X.to_numpy() , Y.to_numpy() , test_size=0.2 , random_state=42 )

    sklearn_model = linear.LogisticRegression(max_iter= 20000)
    my_model = Logistic_regression_model(X_train , Y_train , 0.1 , 20000)


    sklearn_model.fit(X_train , Y_train)
    my_model.fit()

    plot_costs(my_model.costs)

    yhat_sk_model = sklearn_model.predict(X_test)
    yhat_my_model = my_model.predict(X_test)
    sk_accuracy = accuracy_score(Y_test , yhat_sk_model) 
    my_accuracy = accuracy_score(Y_test , yhat_my_model)

    plot_accuracies(sk_accuracy , my_accuracy)
    
    save_model(sklearn_model , "sklearn_model")
    save_model(my_model , "my_model")




if __name__ == "__main__":

    main()