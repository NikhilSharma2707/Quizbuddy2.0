import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
df=pd.read_csv("C:\quizbuddy2.0\src\datasets\preprocessed_dataset.csv")
save_dir="C:\quizbuddy2.0\src\plots"
from wordcloud import WordCloud
df['processed_input'] = df['processed_input'].astype(str)
df['processed_input'].fillna('Unknown', inplace=True)
# Create a word cloud of topics
text = ' '.join(df['processed_input'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.savefig(os.path.join(save_dir, ' Token_Length_plot.png'), dpi=300)
plt.show()
