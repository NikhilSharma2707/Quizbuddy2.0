import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
df=pd.read_csv("C:\quizbuddy2.0\src\datasets\preprocessed_dataset.csv")
save_dir="C:\quizbuddy2.0\src\plots"
df['processed_input'] = df['processed_input'].astype(str)


df['processed_input'].fillna('Unknown', inplace=True)


plt.figure(figsize=(10, 6))
plt.bar(df['processed_input'], df['token_length'], color='skyblue')

plt.xlabel('Topic')
plt.ylabel('Token Length')
plt.title('Token Length Comparison Between Common Topics')
plt.xticks(rotation=90)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, ' Token_Length_plot.png'), dpi=300)
plt.show()
