# Трекинг объектов для компании Ренью

__Задача:__

Необходимо разработать решение для отслеживания и сортировки мусора на конвейере – выделять пластиковые бутылки в общем потоке предметов.

__Метрика:__

Необходимо максимизировать метрику MOTA (Multiple Object Tracking Accuracy), которая вычисляется по формуле:

$$ MOTA = 1 - \frac{\sum_tFN_t+FP_t+IDS_t}{\sum_tGT_t},   $$ 

где FN_t  - количество пропусков на кадре 𝑡, 

   FP_t  - количество ложных срабатываний на кадре 𝑡,
 
   IDS_t - количество несоответствий на кадре 𝑡,
 
   IGT_t - количество размеченных объектов на кадре 𝑡.



__Данные:__

yolov10x_v2_4_best.pt - модель детектора

gt.txt - истинный файл с данными о треке объекта.

labels.txt - файл с наименованием классов для сортировки.

images - папка с фреймами обектов.

test.mp4 - тестовое видео передвижения обьектов на конвейере.

__План работы:__

* Загрузка библиотек.
* Установка параметров проекта.
* Ознакомление с данными.
* Построение baseline решения.
* Обучить трекер с использованием алгоритма трекинга ByteTrack.
* Проанализировать результат, в том числе визуально.
* Подобрать границы трекинга объектов.
* Рассчитать время на обработку одного кадра.


__Технические требования:__
* opencv-python 4.10.0
* pandas 1.2.4
* matplotlib 3.7.5
* Pillow 9.5.0
* scikit-learn 1.2.2
