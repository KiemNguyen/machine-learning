# Upper Confidence Bound

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing Upper Confidence Bound algo
import math
N = 10000
d= 10
ads_selected = []
number_of_selection = [0] * d
sum_of_reward = [0] * d
total_reward = 0

for n in range(0, N):
    max_upper_bound = 0
    ad = 0
    for i in range(0, d):
        if (number_of_selection[i] > 0):
            # Compute average reward of ad i up to round n
            average_reward = sum_of_reward[i] / number_of_selection[i]
            # Compute condidence interval
            delta_i = math.sqrt(3/2 * math.log(n + 1) / number_of_selection[i])
            # Compute upper confidence bounce
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    number_of_selection[ad] = number_of_selection[ad] + 1 
    reward = dataset.values[n, ad]
    sum_of_reward[ad] = sum_of_reward[ad] + reward
    total_reward = total_reward + reward

# Visualising the result
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()