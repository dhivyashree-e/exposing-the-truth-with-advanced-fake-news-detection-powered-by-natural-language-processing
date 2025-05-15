import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample fake and real news dataset
data = {
    'text': [
        'Government confirms new health policy benefits all.',
        'Breaking: Celebrity found alive on Mars!',
        'Elections held peacefully across the country.',
        'Scientists say chocolate cures COVID-19.',
        'Prime Minister visits flood-hit areas.'
    ],
    'label': [1, 0, 1, 0, 1]  # 1 = Real, 0 = Fake
}
df = pd.DataFrame(data)

# Vectorize text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
