# OCR_1

Алгоритм выявления документов, содержащих чувствительную информацию, на основе методов распознавания образов. Предлагаемое решение включает в себя модуль распознавания текста EasyOCR со встроенным средством детектирования текста CRAFT и модуль классификации SVM c TFIDF векторизацией. Представлены экспериментальные результаты, демонстрирующие прирост точности нового алгоритма на 9,8% в сравнении с существующим научнотехническим уровнем.

С текстом статьи исследования, в рамках которого реализовывался алгоритм, вы можете ознакомиться по ссылке ниже:
https://drive.google.com/file/d/1m2osNAVpnLfFU3w-s8Pu-tWMSg4cGvaX/view?usp=sharing

# Установка

Установка easyocr в режиме использования GPU CUDA [1].

Устанавливаем дистрибутив Anaconda Individual Edition [2].

Создаем среду в Anaconda Prompt командой:

conda create –name deeplearning

(https://www.youtube.com/watch?v=2S1dgHpqCdk)

Входим в созданную среду deeplearning командой:

Conda activate deeplearning

Устанавливаем PyTorch [3] в созданной среде командой:

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch (в зависимости от конфигурации)

Устанавливаем последнюю версию графического драйвера для своей конфигурации, в случае исследовательской машины – это NVIDIA GeForce для GeForce GTX 1050 [4].

Устанвливаем CUDA Toolkit 11.3 [5].

Далее заходим в PyCharm и выбираем Python 3.9 в созданной среде deeplearning: Settings/Project: xxx/Python Interpreter/add… Conda Environment/Existing environment 

Для последней стабильной версии используем команду в Anaconda Prompt:

pip install easyocr

Также устанавливаем opencv:

pip install opencv-python==4.5.4.60

Устанавливаем пакет для обработки естественного языка Natural Language Toolkit (NLTK) 

pip install nltk 

1.	https://github.com/JaidedAI/EasyOCR
2.	https://www.anaconda.com/products/individual/download-success-2
3.	https://pytorch.org/get-started/locally/#windows-package-manager
4.	https://www.nvidia.ru/Download/index.aspx?lang=ru
5.	https://developer.nvidia.com/cuda-11.3.0-download-archive

Меняем в программе параметр gpu с False на True

reader = easyocr.Reader(['en'], gpu=True, recog_network='english_g2')

# Как использовать?

Исходное изображение для классификации помещается в папку input проекта. После запуска main.py алгоритм произведет одноклассовое предсказание содержимого
текста документа. Если предсказана метка 1, то документ содержит чувствительную информацию, и выводится оповещение об этом в виде текста ‘sensitive’, иначе выводится ‘non sensitive’.
