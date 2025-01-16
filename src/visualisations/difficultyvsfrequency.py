import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
df=pd.read_csv("C:\quizbuddy2.0\src\datasets\preprocessed_dataset.csv")
save_dir="C:\quizbuddy2.0\src\plots"
topic_difficulty = df.groupby(['processed_input', 'difficulty']).size().unstack(fill_value=0)

# Plot grouped bar chart
topic_difficulty.plot(kind='bar', stacked=False, figsize=(12, 6))
plt.title('Frequency of Topics by Difficulty')
plt.xlabel('Topics')
plt.ylabel('Frequency')
plt.savefig(os.path.join(save_dir, ' Token_Length_plot.png'), dpi=300)

plt.show()