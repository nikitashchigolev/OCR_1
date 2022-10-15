from sklearn.datasets import load_files
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def output(x, y):
    print('Accuracy =', accuracy_score(x, y),
          '\nPrecision =', precision_score(x, y),
          '\nRecall =', recall_score(x, y),
          '\nF1-score =', f1_score(x, y),
          '\nError rate =', (1 - accuracy_score(x, y)))


import seaborn
import matplotlib.pyplot as plt


def plot_confusion_matrix(data, labels, output_filename):
    seaborn.set(color_codes=True)
    plt.figure(1, figsize=(9, 6))

    plt.title("Confusion Matrix")

    seaborn.set(font_scale=1.4)
    ax = seaborn.heatmap(data, annot=True, fmt='g', cbar_kws={'label': 'Scale'})

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set(ylabel="Predicted class", xlabel="Actual class")

    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close()


doc_data = load_files("./train/", encoding="utf-8", decode_error="replace")
X_train, y_train = doc_data.data, doc_data.target

doc_data = load_files("./test/", encoding="utf-8", decode_error="replace")
X_test, y_test = doc_data.data, doc_data.target

# создание конвейеров обучения для моделей
bayes = Pipeline([
    ("count vectorizer", CountVectorizer(stop_words="english", max_features=5000, decode_error="ignore")),
    ("bayes", MultinomialNB())
])
bayes_tfidf = Pipeline([
    ("count vectorizer", TfidfVectorizer(stop_words="english", max_features=5000, decode_error="ignore")),
    ("bayes", MultinomialNB())
])
destree = Pipeline([
    ("count vectorizer", CountVectorizer(stop_words="english", max_features=5000, decode_error="ignore")),
    ("destree", DecisionTreeClassifier())
])
destree_tfidf = Pipeline([
    ("count vectorizer", TfidfVectorizer(stop_words="english", max_features=5000, decode_error="ignore")),
    ("destree", DecisionTreeClassifier())
])
sgd = Pipeline([
    ("count vectorizer", CountVectorizer(stop_words="english", max_features=5000, decode_error="ignore")),
    ("sgd", SGDClassifier(loss="modified_huber"))
])
sgd_tfidf = Pipeline([
    ("tfidf_vectorizer", TfidfVectorizer(stop_words="english", max_features=5000, decode_error="ignore")),
    ("sgd", SGDClassifier())
])
svc = Pipeline([
    ("count_vectorizer", CountVectorizer(stop_words="english", max_features=5000, decode_error="ignore")),
    ("linear svc", LinearSVC())
])
svc_tfidf = Pipeline([
    ("tfidf_vectorizer", TfidfVectorizer(stop_words="english", max_features=5000, decode_error="ignore")),
    ("linear svc", LinearSVC())
])
ranfor = Pipeline([
    ("count_vectorizer", CountVectorizer(stop_words="english", max_features=5000, decode_error="ignore")),
    ("ranfor", RandomForestClassifier())
])
ranfor_tfidf = Pipeline([
    ("tfidf_vectorizer", TfidfVectorizer(stop_words="english", max_features=5000, decode_error="ignore")),
    ("ranfor", RandomForestClassifier())
])
logreg = Pipeline([
    ("count_vectorizer", CountVectorizer(stop_words="english", max_features=5000, decode_error="ignore")),
    ("logreg", LogisticRegression())
])
logreg_tfidf = Pipeline([
    ("tfidf_vectorizer", TfidfVectorizer(stop_words="english", max_features=5000, decode_error="ignore")),
    ("logreg", LogisticRegression())
])
knn = Pipeline([
    ("count_vectorizer", CountVectorizer(stop_words="english", max_features=5000, decode_error="ignore")),
    ("knn", KNeighborsClassifier())
])
knn_tfidf = Pipeline([
    ("tfidf_vectorizer", TfidfVectorizer(stop_words="english", max_features=5000, decode_error="ignore")),
    ("knn", KNeighborsClassifier())
])

all_models = [
    ("bayes", bayes),
    ("bayes_tfidf", bayes_tfidf),
    ("destree", destree),
    ("destree_tfidf", destree_tfidf),
    ("sgd", sgd),
    ("sgd_tfidf", sgd_tfidf),
    ("svc", svc),
    ("svc_tfidf", svc_tfidf),
    ("ranfor", ranfor),
    ("ranfor_tfidf", ranfor_tfidf),
    ("logreg", logreg),
    ("logreg_tfidf", logreg_tfidf),
    ("knn", knn),
    ("knn_tfidf", knn)
]

# нахождение самой производительной модели
unsorted_scores = [(name, cross_val_score(model, X_train, y_train, cv=None).mean()) for name, model in all_models]
scores = sorted(unsorted_scores, key=lambda x: -x[1])
print(scores)

model = svc_tfidf
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
output(y_test, y_pred)

tn, fp, fn, tp = confusion_matrix(list(y_test), list(y_pred), labels=[0, 1]).ravel()

data = [[tp, fp],
        [fn, tn]]
labels = ["private", "public"]

# создание матрицы несоответствий
plot_confusion_matrix(data, labels, "confusion_matrix3.png")