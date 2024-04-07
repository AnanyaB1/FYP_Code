import os
import pandas as pd
import pandas as pd
import time
# import r2 score
from sklearn.metrics import r2_score

scores = ["CogFluidComp", "PicSeq", "PicVocab", "ReadEng", "CardSort", "ListSort", "Flanker", "ProcSpeed"]  # Removed duplicate "CardSort"
scores = sorted(scores)

# base_directory = "/data/ananya012/results" # to change
base_directory = "/data/ananya012/results/MV_GCN"

# Function to calculate the mean of a list
def calculate_mean(values):
    return sum(values) / len(values) if values else None

# Iterate over each score
for score in scores:
    print(f"Processing score: {score}")
    directory = os.path.join(base_directory, score)

    # Lists to hold all the values of corr, rmse, and mae for the current score
    corrs = []
    rmses = []
    maes = []
    all_r2 = []

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
                    corrs.append(data.iloc[1, 1])  # Append corr value
                    rmses.append(data.iloc[2, 1])  # Append rmse value
                    maes.append(data.iloc[3, 1])  # Append mae value
                except Exception as e:
                    print(f"An error occurred while reading {final_csv_path}: {e}")
        start = "predict_train"
        end = ".csv"
        for i in range(1, 6):
            train_res_oath = os.path.join(root, dir_name, start + str(i) + end)
            if os.path.isfile(train_res_oath):
                data = pd.read_csv(train_res_oath)
                predicted = data['Predicted']
                actual = data['Actual']
                r2 = r2_score(actual, predicted)
                # print(r2)
                all_r2.append(r2)
            






    # Assuming corrs, rmses, and maes are your lists with collected values
    corrs_series = pd.Series(corrs)
    rmses_series = pd.Series(rmses)
    maes_series = pd.Series(maes)
    r2_series = pd.Series(all_r2)

    # Calculate the mean for each
    mean_corr = corrs_series.mean()
    mean_rmse = rmses_series.mean()
    mean_mae = maes_series.mean()
    mean_r2 = r2_series.mean()

    # Calculate the standard deviation for each
    std_corr = corrs_series.std()
    std_rmse = rmses_series.std()
    std_mae = maes_series.std()
    std_r2 = r2_series.std()

    # Print out the averages for the current score
    print(f"Mean Correlation for {score}: {round(mean_corr, 2)} ({round(std_corr, 2)})")
    print(f"Mean RMSE for {score}: {round(mean_rmse, 2)} ({round(std_rmse, 2)})")
    print(f"Mean MAE for {score}: {round(mean_mae, 2)} ({round(std_mae, 2)})")
    print(f"Mean R2 for {score}: {round(mean_r2, 2)} ({round(std_r2, 2)})")
    # print in "corr (std) & rmse (std) & mae (std) & r2 (std) to 2 dp"
    print(f"{mean_corr:.2f} ({std_corr:.2f}) & {mean_rmse:.2f} ({std_rmse:.2f}) & {mean_mae:.2f} ({std_mae:.2f}) & {mean_r2:.2f} ({std_r2:.2f})")
    print()
 
