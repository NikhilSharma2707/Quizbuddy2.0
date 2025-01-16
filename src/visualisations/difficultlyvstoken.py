import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
df=pd.read_csv("C:\quizbuddy2.0\src\datasets\preprocessed_dataset.csv")
save_dir="C:\quizbuddy2.0\src\plots"

sns.boxplot(x='difficulty', y='token_length', data=df, palette='Set2')
plt.title('Token Length vs Difficulty')
plt.xlabel('Difficulty')
plt.ylabel('Token Length')
plt.show()

# Violin plot (if you prefer a more detailed visualization)
sns.violinplot(x='difficulty', y='token_length', data=df, palette='Pastel1')
plt.title('Token Length vs Difficulty')
plt.xlabel('Difficulty')
plt.ylabel('Token Length')
plt.show()