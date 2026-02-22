from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def run_baseline(train_texts, train_labels, test_texts, test_labels):
    vectorizer = TfidfVectorizer(max_features=20000)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, train_labels)

    preds = model.predict(X_test)
    print("\nBaseline Results:\n")
    print(classification_report(test_labels, preds))
