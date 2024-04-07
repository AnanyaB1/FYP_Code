import os
import pandas as pd
import pandas as pd
import time
from sklearn.metrics import r2_score

scores = ["CogFluidComp", "PicSeq", "PicVocab", "ReadEng", "CardSort", "ListSort", "Flanker", "ProcSpeed"]  # Removed duplicate "CardSort"
scores = sorted(scores)

for score in scores:
    print(score, end=", ")
base_directory = "/data/ananya012/results/GCN" # to change

# Function to calculate the mean of a list
def calculate_mean(values):
    return sum(values) / len(values) if values else None

# Iterate over each score

for score in scores:
    print(f"Processing score: {score}")
    connectome = ['FC', 'SC']
    secondary_directory = os.path.join(base_directory, score)
    for conn in connectome:
        directory = os.path.join(secondary_directory, conn)

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
        print(f"Mean Correlation for {score} {conn}: {round(mean_corr, 2)} ({round(std_corr, 2)})")
        print(f"Mean RMSE for {score} {conn}: {round(mean_rmse, 2)} ({round(std_rmse, 2)})")
        print(f"Mean MAE for {score} {conn}: {round(mean_mae, 2)} ({round(std_mae, 2)})")
        print(f"Mean R2 for {score} {conn}: {round(mean_r2, 2)} ({round(std_r2, 2)})")
        # print in latex table format, corr (std) & rmse (std) & mae (std) & r^2 (std) all print to minimum 2 dp
        
        print(f"{round(mean_corr, 2):.2f} ({round(std_corr, 2):.2f}) & {round(mean_rmse, 2):.2f} ({round(std_rmse, 2):.2f}) & {round(mean_mae, 2):.2f} ({round(std_mae, 2):.2f}) & {round(mean_r2, 2):.2f} ({round(std_r2, 2):.2f})")
        print()

