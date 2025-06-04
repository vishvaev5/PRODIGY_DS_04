import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


train_df = pd.read_csv("D:/Python/PRODIGY_DS_04/twitter_training.csv", header=None, names=['ID', 'Entity', 'Sentiment', 'Text'])

train_df = train_df[train_df['Sentiment'].isin(['Positive', 'Negative', 'Neutral'])]

train_df = train_df.dropna(subset=['Text', 'Sentiment'])

X = train_df['Text']
y = train_df['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)


plt.figure(figsize=(8,6))
sns.countplot(data=train_df, x='Sentiment', hue='Sentiment', palette='pastel', order=train_df['Sentiment'].value_counts().index, legend=False)
plt.title('Sentiment Distribution in Training Data')
plt.xlabel('Sentiment')
plt.ylabel('Tweet Count')
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
sns.countplot(data=train_df, x='Entity', hue='Sentiment', palette='Set2', order=train_df['Entity'].value_counts().index[:15])
plt.title('Sentiment Distribution Across Top 15 Entities')
plt.ylabel('Tweet Count')
plt.xlabel('Entity')
plt.xticks(rotation=45)
plt.legend(title='Sentiment')
plt.tight_layout()
plt.show()

print("\nSentiment Analysis on Twitter Data is Completed by Arun Balaji! ")