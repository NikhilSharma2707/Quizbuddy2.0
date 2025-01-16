import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
df=pd.read_csv("C:\quizbuddy2.0\src\datasets\preprocessed_dataset.csv")
save_dir="C:\quizbuddy2.0\src\plots"
df['processed_input'] = df['processed_input'].astype(str)


df['processed_input'].fillna('Unknown', inplace=True)
plt.figure(figsize=(10, 6))
plt.hist(df['token_length'], bins=10, color='skyblue', edgecolor='black')
plt.title('Token Length Distribution')
plt.xlabel('Token Length')
plt.ylabel('Frequency')
plt.savefig('token_length_distribution.png', dpi=300)

# Second Plot
plt.figure(figsize=(10, 6))
plt.bar(df['processed_input'], df['token_length'], color='orange')
plt.title('Token Length by Processed Input')
plt.xlabel('Processed Input')
plt.ylabel('Token Length')
plt.xticks(rotation=45)
plt.savefig('token_length_by_processed_input.png', dpi=300)
plt.savefig(os.path.join(save_dir, 'multiple.png'), dpi=300)
plt.show()