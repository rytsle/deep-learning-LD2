# VARIANTAI
# DUOMENU RINKINYS NR. 2
# ARCHITEKTUROS 4; 5; 8.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# 1. Duomenų nuskaitymas
labels_data = pd.read_csv('LD2_dataset/labels.csv')

print(labels_data.head())