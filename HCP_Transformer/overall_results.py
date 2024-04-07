import os
import numpy as np
import pandas as pd

def get_metrics_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df

def aggregate_metrics(parent_folder, MHSA):
    metrics_files = []
    for root, dirs, files in os.walk(parent_folder):
        if MHSA:
            if 'NoMHSA' not in root:
                for file in files:
                    if file.startswith('metrics_result_fold_') and file.endswith('.csv'):
                        metrics_files.append(os.path.join(root, file))
        else:
            if 'NoMHSA' in root:
                for file in files:
                    if file.startswith('metrics_result_fold_') and file.endswith('.csv'):
                        metrics_files.append(os.path.join(root, file))
    # print(metrics_files)
    metrics_list = []
    for file_path in metrics_files:
        metrics_df = get_metrics_from_csv(file_path)
        metrics_list.append(metrics_df)
    
    combined_metrics_df = pd.concat(metrics_list)
    mean_metrics = combined_metrics_df.mean()
    std_metrics = combined_metrics_df.std()
    
    return mean_metrics, std_metrics

def print_metrics(mean_metrics, std_metrics, score):
    """Print metrics in the specified formats."""
    print(f"Mean Correlation for {score}: {mean_metrics['Test Corr']:.2f} ({std_metrics['Test Corr']:.2f})")
    print(f"Mean RMSE for {score}: {mean_metrics['Test RMSE']:.2f} ({std_metrics['Test RMSE']:.2f})")
    print(f"Mean MAE for {score}: {mean_metrics['Test MAE']:.2f} ({std_metrics['Test MAE']:.2f})")
    print(f"Mean R2 for {score}: {mean_metrics['Test R2']:.2f} ({std_metrics['Test R2']:.2f})")
    
    print(f"{mean_metrics['Test Corr']:.2f} ({std_metrics['Test Corr']:.2f}) & {mean_metrics['Test RMSE']:.2f} ({std_metrics['Test RMSE']:.2f}) & {mean_metrics['Test MAE']:.2f} ({std_metrics['Test MAE']:.2f}) & {mean_metrics['Test R2']:.2f} ({std_metrics['Test R2']:.2f})")

scores = ["CogFluidComp", "PicSeq", "PicVocab", "ReadEng", "CardSort", "ListSort", "Flanker", "ProcSpeed"]

# for score in scores:
# score = "Flanker"
score = "ProcSpeed"
parent_folder = f'/home/ananya012/HCP_Transformer/Data/Train_Test_Only/{score}/'
# if not os.path.exists(parent_folder):
#     continue

print("WITH MSHA")
mean_metrics, std_metrics = aggregate_metrics(parent_folder, True)
print_metrics(mean_metrics, std_metrics, score)


# print("WITHOUT MSHA")
# mean_metrics, std_metrics = aggregate_metrics(parent_folder, False)
# print_metrics(mean_metrics, std_metrics, score)