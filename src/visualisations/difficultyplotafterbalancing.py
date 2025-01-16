import pandas as pd
import matplotlib.pyplot as plt
import os
save_dir="C:\quizbuddy2.0\src\plots"
# Bar plot for the count of each difficulty level
df=pd.read_csv(r"C:\quizbuddy2.0\src\datasets\balanced\final.csv")
df['difficulty'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Distribution of Difficulty Levels After Balancing The Data')
plt.xlabel('Difficulty Level')
plt.ylabel('Count')
plt.show()

# Alternatively, a pie chart
df['difficulty'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99'])
plt.title('Proportion of Difficulty Levels')

plt.savefig(os.path.join(save_dir, 'difficultyafterbalance_proportion_plot.png'), dpi=300)
plt.show()