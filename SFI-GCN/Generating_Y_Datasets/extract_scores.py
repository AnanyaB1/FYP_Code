import pandas as pd
import numpy as np

hcp_metadata = pd.read_csv('/home/ananya012/SC-FC-fusion/gae/HCP_metadata.csv')

with open('/home/ananya012/SC-FC-fusion/gae/overlap_HCP.txt', 'r') as file:
    overlap_subjects = file.read().splitlines()
    # Convert the list of strings to a list of integers if necessary
    overlap_subjects = [int(subject) for subject in overlap_subjects]


filtered_hcp_metadata = hcp_metadata[hcp_metadata['Subject'].isin(overlap_subjects)]


print("overlap subjects: ", len(overlap_subjects), len(filtered_hcp_metadata['Subject'].unique()))

gender_mapped = filtered_hcp_metadata['Gender'].map({'M': 0, 'F': 1})

# filtered_hcp_metadata.to_csv('filtered_HCP_metadata.csv', index=False)

Gender_HCP = gender_mapped.values
Age_HCP = filtered_hcp_metadata["Age"].values
PicSeq_Unadj_HCP = filtered_hcp_metadata["PicSeq_Unadj"].values
CardSort_Unadj_HCP = filtered_hcp_metadata["CardSort_Unadj"].values
ReadEng_Unadj_HCP = filtered_hcp_metadata["ReadEng_Unadj"].values
PicVocab_Unadj_HCP = filtered_hcp_metadata["PicVocab_Unadj"].values
ProcSpeed_Unadj_HCP = filtered_hcp_metadata["ProcSpeed_Unadj"].values
ListSort_Unadj_HCP = filtered_hcp_metadata["ListSort_Unadj"].values
Flanker_Unadj_HCP = filtered_hcp_metadata["Flanker_Unadj"].values


# np.save('scores/Gender_HCP.npy', Gender_HCP)
np.save('scores/Age_HCP.npy', Age_HCP)
np.save('scores/PicSeq_Unadj_HCP.npy', PicSeq_Unadj_HCP)
np.save('scores/CardSort_Unadj_HCP.npy', CardSort_Unadj_HCP)
np.save('scores/ReadEng_Unadj_HCP.npy', ReadEng_Unadj_HCP)
np.save('scores/PicVocab_Unadj_HCP.npy', PicVocab_Unadj_HCP)
np.save('scores/ProcSpeed_Unadj_HCP.npy', ProcSpeed_Unadj_HCP)
np.save('scores/ListSort_Unadj_HCP.npy', ListSort_Unadj_HCP)
np.save('scores/Flanker_Unadj_HCP.npy', Flanker_Unadj_HCP)