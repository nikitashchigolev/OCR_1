import torch
torch.cuda.is_available()

import easyocr
import cv2
import glob
import nltk

path = ["private/*.*", "public/*.*"]
reader = easyocr.Reader(['en'], gpu=False, recog_network='english_g2')

with open("sensitive words.txt") as file:
    senswords = [row.strip() for row in file]
expresults = [[],[]] #датафрейм: файл, классификация для приватных и публичных документов
data = [[]]

i = 0
for i in range(2):
    img_number = 1
    #алгоритм распознавания и извлечения текста на основе easyocr с добавление извлеченной информации в датафрейм
    for file in glob.glob(path[i]):
        print(file)
        expresults[0].append(file)
        img = cv2.imread(file, 0)
        results = reader.readtext(img, detail=0, paragraph=False)
        count = 0
        while count < len(results):
            data[0].append((results[count].lower()).split()) #перевод символов в нижний регистр для получения меньшего расстояния левенштейна
            count +=1                                        #и разделение строки на списки отдельных слов для их последующего использования в
        count += 1                                           #алгоритме классификации
        data[0].append(len(results)) #добавление количества элементов списка

    #алгоритм классификации документов на основе споттинга чувствительных слов
        cf = -1
        sim = 0
        j = 0
        k = 0
        m = 0
        for j in (range(data[0][-1])):
            for m in range(len(data[0][j])):
                for k in range(len(senswords)):
                    sim = 1 / (nltk.edit_distance(data[0][j][m], senswords[k]) + 1)
                    if cf < sim:
                        cf = sim
                    k += 1
                m += 1
            j += 1
        j = 0
        k = 0
        m = 0
        if cf > 0.8:
            expresults[1].append('sensitive')
        else:
            expresults[1].append('non sensitive')
        cf = -1
        sim = 0

        img_number +=1
        data[0].clear()

#оценка классификатора
tp = 0
tn = 0
fp = 0
fn = 0

i = 0
while i < 350:
    if expresults[1][i] == 'sensitive':
        tp +=1
    else:
        fp +=1
    i +=1
while i < 700:
    if expresults[1][i] == 'non sensitive':
        tn +=1
    else:
        fn +=1
    i += 1

accuracy = (tp+tn)/(tp+tn+fp+fn)
recall = tp/(tp+fn)
precision = tp/(tp+fp)
f1score = 2*((precision*recall)/(precision+recall))

print('TP = ',tp,'\nTN = ',tn,'\nFP = ',fp,'\nFN = ',fn,'\n')
print('Accuracy = ', accuracy,'\nRecall = ',recall,'\nPrecision = ',precision,'\nF1 score = ',f1score)

