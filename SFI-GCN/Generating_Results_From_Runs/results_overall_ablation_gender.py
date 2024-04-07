import os
import pandas as pd
import pandas as pd
import time
from sklearn.metrics import r2_score

base_directory = "/data/ananya012/results/GCN" # to change
# "/data/ananya012/results/MV_GCN"

# Function to calculate the mean of a list
def calculate_mean(values):
    return sum(values) / len(values) if values else None

score = "Gender"

connectome = ['FC', 'SC']
secondary_directory = os.path.join(base_directory, score)
for conn in connectome:
    directory = os.path.join(secondary_directory, conn)

    # Lists to hold all the values of corr, rmse, and mae for the current score
    accus = []
    # Iterate over each file in the directory
    for root, dirs, files in os.walk(directory):
        # print("here", root, dirs, files)
        for dir_name in dirs:
            # Construct the path to the final.csv file
            final_csv_path = os.path.join(root, dir_name, 'final_result.csv')
            if os.path.isfile(final_csv_path):
                # Read the final.csv file
                try:
                    data = pd.read_csv(final_csv_path, header=None)
                    # Assuming the csv has one row with corr, rmse, mae in this order
                    # print(data[1][1])
                    accus.append(data.iloc[1, 1])  # Append corr value
                except Exception as e:
                    print(f"An error occurred while reading {final_csv_path}: {e}")

    # calc accuracy mean and std
    # convert to percentage
    accus = [float(accu) * 100 for accu in accus]
    mean_accu = calculate_mean(accus)
    std_accu = pd.Series(accus).std()

    # Print out the averages for the current score
    print(f"Mean Accuracy for {score} {conn}: {round(mean_accu, 2)} ({round(std_accu, 2)})")
