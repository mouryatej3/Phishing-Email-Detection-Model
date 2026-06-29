import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# Load Dataset

data = pd.read_csv("dataset.csv")


X = data["email"]
y = data["label"]


# Split data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)



# Machine Learning Model

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english"
        )
    ),

    (
        "classifier",
        MultinomialNB()
    )
])


# Training

model.fit(
    X_train,
    y_train
)



# Prediction

prediction = model.predict(X_test)



# Accuracy

accuracy = accuracy_score(
    y_test,
    prediction
)


print(
    "\nModel Accuracy:",
    accuracy*100,
    "%"
)



# Confusion Matrix

cm = confusion_matrix(
    y_test,
    prediction
)


print("\nConfusion Matrix")
print(cm)



plt.figure(figsize=(5,5))

plt.imshow(cm)

plt.title(
    "Phishing Email Detection Confusion Matrix"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.colorbar()

plt.savefig(
    "confusion_matrix.png"
)

plt.show()



print(
    "\nClassification Report"
)

print(
    classification_report(
        y_test,
        prediction
    )
)



# Test Custom Email

while True:

    email = input(
        "\nEnter Email Text: "
    )


    result = model.predict(
        [email]
    )


    if result[0] == "phishing":
        print(
            "⚠️ Result: PHISHING EMAIL"
        )

    else:
        print(
            "✓ Result: SAFE EMAIL"
        )
