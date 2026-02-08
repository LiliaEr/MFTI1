# MFTI1 — Учебный проект по распознаванию лиц

Проект реализует полный пайплайн Face Recognition:

1. Отбор изображений  
2. Сборка датасета  
3. Кроп лиц  
4. Alignment лиц по ключевым точкам (Stacked Hourglass)  
5. Обучение модели распознавания (CE / ArcFace)  
6. Проверка качества (verification, IR@k)  
7. Дообучение Triplet loss  

**Важно:**  
Прошу понять и простить, так как часть работы проводилась локально, а часть в Colab, в путях отсутствие единой структуры.
В следующий раз я очень постараюсь заранее продумать этот момент, но сейчас баюсь что либо трогать, чтобы не сломалось.

---

## 1. Что находится в GitHub

В репозитории находятся **только код и ноутбуки**:

MFTI1/
│
├── cv/
├── 1/		# папки 1-4 с фото для проверки работы pipline (06_verify_random.ipynb)
├── 2/
├── 3/
├── 4/
│
├── csv/
│ ├── train_align.csv
│ ├── train_align_112.csv
│ ├── train_recognition_112.csv
│ ├── val_align.csv
│ ├── val_align_112.csv
│ ├── val_recognition_112.csv
│ ├── test_align.csv
│ ├── test_align_112.csv
│ └── test_recognition_112.csv
│
├── 01_photo_selection.ipynb
├── 02_build_dataset.py
├── 03_crop.ipynb
├── 04_Hourglass.ipynb
├── 05_CE_ArcFace.ipynb
├── 06_verify_random.ipynb
├── 07_ir_metric_ipynb.ipynb
├── 08_triplet.ipynb
│
├── face_pipeline.py
├── pipeline_core.py
├── README.md
└── requirements.txt


---

## 2. Что НЕ находится в GitHub (скачивается отдельно)

Следующие папки **большие по размеру**, поэтому они **не загружаются в GitHub**  
и доступны по **ссылке на Google Drive**:

arcface_mainset_1/
│
├── cropped/  
├── aligned_112/
└── checkpoints/


 **Ссылка на Google Drive для arcface_mainset_1:**  
https://drive.google.com/drive/folders/1FaW2NIQy_sJGdbMVDS04KfEIDngT8iDa?usp=sharing
---

## 3. Куда положить данные 

### Вариант 1 — Google Colab (рекомендуется для части ноутбуков)

После подключения Google Drive структура должна выглядеть так:

/content/drive/MyDrive/Colab Notebooks/MFTI/
└── arcface_mainset_1/
├── cropped/
│ ├── train/
│ ├── val/
│ └── test/
│
├── aligned_112/
│ ├── train/
│ ├── val/
│ └── test/
│
├── checkpoints/
│ ├── stacked_hourglass_best.pt
│ ├── ce_best_700.pt
│ ├── arcface_best.pt
│ ├── arcface_best_fr.pt
│ └── triplet_best.pt
│
└── splits_compact/
├── train_align.csv
├── train_align_112.csv
├── train_recognition_112.csv
├── val_align.csv
├── val_align_112.csv
├── val_recognition_112.csv
├── test_align.csv
├── test_align_112.csv
└── test_recognition_112.csv


Именно этот путь используется в ноутбуках 04_Hourglass.ipynb, 05_CE_ArcFace.ipynb, 07_ir_metric_ipynb.ipynb, 08_triplet.ipynb

Ноутбуки 01_photo_selection.ipynb, 02_build_dataset.py, 03_crop.ipynb, 06_verify_random.ipynb (+face_pipeline.py, pipeline_core.py)
запускались локально. Для 06_verify_random.ipynb stacked_hourglass_best.pt, arcface_best_fr.pt", ce_best_700.pt, а также папки 1, 2, 3, 4 должны лежать там же, где запускаемый ноутбук).



---

## 4. Назначение каждого ноутбука и используемые пути

### 01_photo_selection.ipynb  
**Задача:**  
Предварительный анализ и отбор изображений для датасета.

**Использует:**  
- исходные изображения (локальный источник)

**Создаёт:**  
- CSV со списком выбранных изображений

Используется только как подготовительный этап.

---

### 02_build_dataset.py  
**Задача:**  
Формирование структуры датасета вида:

label/
├── img1.jpg
├── img2.jpg


**Использует:**  
- CSV с выбранными изображениями  
- исходный архив изображений

**Создаёт:**  
- физическую структуру датасета

Запускается локально **один раз**, далее не требуется.

---

### 03_crop.ipynb  
**Задача:**  
Кроп лиц из исходных изображений.

**Читает:**  
arcface_mainset_1/


**Создаёт:**  
arcface_mainset_1/cropped/
├── train/
├── val/
└── test/


А также CSV-файлы:
splits_compact/*_align.csv


---

### 04_Hourglass.ipynb  
**Задача:**  
Alignment лиц по ключевым точкам с помощью **Stacked Hourglass Network**.

**Читает:**  
arcface_mainset_1/cropped/
splits_compact/*_align.csv


**Создаёт:**  
arcface_mainset_1/aligned_112/
arcface_mainset_1/checkpoints/stacked_hourglass_best.pt
splits_compact/*_align_112.csv


---

### 05_CE_ArcFace.ipynb  
**Задача:**  
Обучение моделей распознавания лиц:

- CE baseline  
- ArcFace

**Читает:**  
aligned_112/
splits_compact/*_recognition_112.csv


**Создаёт:**  
checkpoints/
├── ce_best_700.pt
├── arcface_best.pt
└── arcface_best_fr.pt


---

### 06_verify_random.ipynb  
**Задача:**  
Демонстрационная проверка:  
случайный запрос → top-K совпадений.

**Читает:**  
checkpoints/arcface_best_fr.pt

Используется для визуальной проверки качества.

---

### 07_ir_metric_ipynb.ipynb  
**Задача:**  
Подсчёт метрик качества:

- IR@1, IR@5, IR@10  
- ROC AUC  
- TAR@FAR

**Читает:**  
aligned_112/
splits_compact/
checkpoints/*.pt


**Создаёт:**  
- таблицы метрик  

---

### 08_triplet.ipynb  
**Задача:**  
Дообучение эмбеддингов с использованием **Triplet loss**.

**Читает:**  
aligned_112/
splits_compact/
checkpoints/ce_best_700.pt


**Создаёт:**  
checkpoints/triplet_best.pt
splits_compact/triplet_best_thr.json


---

## 5. Рекомендуемый порядок просмотра 

Если данные уже скачаны:

1. `04_Hourglass.ipynb` — alignment  
2. `05_CE_ArcFace.ipynb` — обучение  
3. `06_verify_random.ipynb` — визуальная проверка  
4. `07_ir_metric_ipynb.ipynb` — метрики  
5. `08_triplet.ipynb` — metric learning  

---

## 6. Итог

Проект демонстрирует полный пайплайн Face Recognition:
от подготовки данных до оценки качества и metric learning.

