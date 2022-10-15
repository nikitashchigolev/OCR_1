import torch
torch.cuda.is_available()

import easyocr
import cv2
import glob
import re

reader = easyocr.Reader(['en'], gpu=False, recog_network='english_g2')

path = ["input/*.*"]
data = [[]]
exttext = ""

#алгоритм обнаружения и извлечения текста с помощью EasyOCR
i = 0
for file in glob.glob(path[i]):
    print(file)
    img = cv2.imread(file, 0)
    #извлечение текста из изображения, получения списка распознанных строк
    results = reader.readtext(img, detail=0, paragraph=False)
    count = 0
    while count < len(results):
        #разделение распознанной строки на отдельные токены, перевод в нижний регистр
        data[0].append((results[count].lower()).split())
        count +=1
    count +=1
    data[0].append(len(results))

    j = 0
    k = 0
    m = 0
    word = ''
    lenword = 0
    #окончательная токенизация извлеченного текста
    for j in (range(data[0][-1])):
        for m in range(len(data[0][j])):
            #удаление специальных символов
            word = re.sub(r"[^a-z]","",data[0][j][m])
            if word != "":
                exttext = exttext + " " + word
            m += 1
        j += 1
    j = 0
    m = 0

text = re.sub(r'.', '', exttext, count = 1)

#алгоритм классификации извлеченной информации с помощью метода опорных векторов
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_files
from sklearn.svm import LinearSVC

doc_data = load_files("./train/", encoding="utf-8", decode_error="replace")
X_train, y_train = doc_data.data, doc_data.target

X_test = [text]
y_test = [0, 1]

#векторизация
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, decode_error="ignore")
vectorizer.fit(X_train)

#создание и обучение классификатора
svc = LinearSVC()
svc.fit(vectorizer.transform(X_train), y_train)

#одноклассовое предсказание
y_pred = svc.predict(vectorizer.transform(X_test))

if y_pred == 1:
    print('sensitive')
else:
    print('non sensitive')