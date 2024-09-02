import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Loading data and creating 3s intervals
normal_data = pd.read_csv('clean_normal_data.csv')
normal_data['interval'] = (normal_data['SECONDS'] // 3).astype(int)

driving_data = pd.read_csv('clean_driving_data.csv')
driving_data['interval'] = (driving_data['SECONDS'] // 3).astype(int)

# Makes each PID a column, makes calculations easier down the line
pivoted_data = normal_data.pivot_table(index='interval', columns='PID', values='VALUE', aggfunc='mean')
driving_pivoted_data = driving_data.pivot_table(index='interval', columns='PID', values='VALUE', aggfunc='mean')

# Calculate Z-Scores for each PID
z_scores = pivoted_data.apply(lambda x: (x - x.mean()) / x.std())
driving_z_scores = driving_pivoted_data.apply(lambda x: (x - x.mean()) / x.std())

# Apply weights (some PID's matter more than others)
# CHECK: This is subjective to my interpretation -- might require some more manual analysis of data/thought

# Speed is 0.55 because it can fluctuate heavily off a stoplight--do not want to misrepresent what aggressive driving is
# Acceleration is 0.7 because heavy fluctuation in acceleration represents aggressive speeding/braking

# Engine RPM can be misrepresented by a gear shift, and the horsepower metric is based on fuel consumption which is not necessarily
# An accurate representation of aggressive driving
weights = {
    'Vehicle speed': 0.55,
    'Vehicle acceleration': 0.7,
    'Engine RPM': 0.4,
    'Horsepower': 0.2
}

weighted_z_scores = z_scores.copy()
driving_weighted_z_scores = driving_z_scores.copy()

for col in weights.keys():
    if col in weighted_z_scores.columns:
        weighted_z_scores[col] = weighted_z_scores[col] * weights[col]

    if col in driving_weighted_z_scores.columns:
        driving_weighted_z_scores[col] = driving_weighted_z_scores[col] * weights[col]

# Sum the weighted Z-Scores to get a final score
# This is the metric that we will "judge/classify" other 3s samples with in the future
pivoted_data['final_score'] = weighted_z_scores.sum(axis=1)
driving_pivoted_data['final_score'] = driving_weighted_z_scores.sum(axis=1)

# Calculate mean and standard deviation of the final scores
# ONLY for the NORMAL DRIVING DATA
mean_score = pivoted_data['final_score'].mean()
std_dev_score = pivoted_data['final_score'].std()

# Classify the scores
# .where() works like a ternary operator
pivoted_data['classification'] = np.where(
    pivoted_data['final_score'] > mean_score + std_dev_score, 'Aggressive',
    np.where(pivoted_data['final_score'] < mean_score - std_dev_score, 'Slow', 'Normal')
)

# Plot the final scores over time to visualize data
# Helps because most of the "NORMAL" data should fall under the normal range (+/- 1 std dev from the mean of this dataset)
plt.figure(figsize=(14, 7))
plt.plot(pivoted_data.index * 3, pivoted_data['final_score'], label='Driving Score', color='blue')
plt.axhline(mean_score, color='green', linestyle='--', label='Mean')
plt.axhline(mean_score + std_dev_score, color='red', linestyle='--', label='+1 Std Dev (Aggressive)')
plt.axhline(mean_score - std_dev_score, color='orange', linestyle='--', label='-1 Std Dev (Slow)')
plt.xlabel('Time (s)')
plt.ylabel('Driving Score')
plt.title('Driving Behavior Over Time')
plt.legend()
plt.show()

## FROM HERE: Classifying the driving data that isn't 'normal', contains more data and should help the model be more accurate
## The purpose of having the "normal" data was to find a "line" to classify samples

driving_pivoted_data['classification'] = np.where(
    driving_pivoted_data['final_score'] > mean_score + std_dev_score, 'Aggressive',
    np.where(driving_pivoted_data['final_score'] < mean_score - std_dev_score, 'Slow', 'Normal')
)

# Plot the final scores over time to visualize data (for the driving dataset)
plt.figure(figsize=(14, 7))
plt.plot(driving_pivoted_data.index * 3, driving_pivoted_data['final_score'], label='Driving Score', color='blue')
plt.axhline(mean_score, color='green', linestyle='--', label='Mean')
plt.axhline(mean_score + std_dev_score, color='red', linestyle='--', label='+1 Std Dev (Aggressive)')
plt.axhline(mean_score - std_dev_score, color='orange', linestyle='--', label='-1 Std Dev (Slow)')
plt.xlabel('Time (s)')
plt.ylabel('Driving Score')
plt.title('Driving Behavior Over Time')
plt.legend()
plt.show()

# Map the classification back to driving_data & save as a new CSV for model training
driving_data['classification'] = driving_data['interval'].map(driving_pivoted_data['classification'])
driving_data.to_csv('classified_driving_data.csv', index=False)