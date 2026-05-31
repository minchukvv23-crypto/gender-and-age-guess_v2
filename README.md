# Курсовой проект: предсказание пола и возраста автора Telegram-текста

**Автор:** Минчук Вадим Владимирович  
**Группа:** СДП-КБ-231  
**Специальность:** Кибербезопасность  

## Описание проекта

Проект предназначен для учебной демонстрации автоматизированного профилирования автора по тексту. Программа обучает нейронную сеть определять объединённый класс автора по короткому Telegram-тексту: пол и возрастная группа.

В проекте используется синтетический датасет. Он не содержит реальные сообщения, имена пользователей, номера телефонов, идентификаторы Telegram или другие персональные данные.

## Основные возможности

- генерация синтетического датасета Telegram-текстов;
- обучение модели MLP на TF-IDF признаках;
- сохранение обученной модели, векторизатора и кодировщика меток;
- построение отчёта классификации, матрицы ошибок и графика обучения;
- проверка модели на новом пользовательском тексте.

## Требования

Для запуска требуется:

- Python 3.10 или новее;
- pip;
- библиотеки из файла `requirements.txt`.

## Структура проекта

```text
coursework_author_profiling/
├── README.md                              # Инструкция по запуску проекта
├── requirements.txt                       # Список зависимостей Python
├── generate_dataset.py                    # Генерация синтетического датасета
├── train_model.py                         # Обучение нейронной сети
├── predict_author_profile.py              # Предсказание класса для нового текста
├── create_report_assets.py                # Создание изображений для отчёта
├── telegram_author_profile_synthetic.csv  # Готовый синтетический датасет
├── author_profile_mlp.pt                  # Сохранённые веса модели
├── tfidf_vectorizer.joblib                # Сохранённый TF-IDF векторизатор
├── label_encoder.joblib                   # Сохранённый кодировщик классов
├── classification_report.txt              # Текстовый отчёт по метрикам
├── classification_report.json             # Отчёт по метрикам в JSON
├── confusion_matrix.csv                   # Матрица ошибок в CSV
├── confusion_matrix.png                   # Изображение матрицы ошибок
├── training_curve.png                     # График обучения
├── training_history.csv                   # История обучения
└── report_assets/                         # Изображения для пояснительной записки
```

## Как запустить проект

### 1. Клонировать репозиторий

```bash
git clone https://github.com/username/coursework_author_profiling.git
cd coursework_author_profiling
```

Если проект уже скачан архивом, достаточно открыть терминал в папке `coursework_author_profiling`.

### 2. Создать и активировать виртуальное окружение

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Установить зависимости

```bash
pip install -r requirements.txt
```

### 4. Быстро проверить уже обученную модель

В архиве уже есть обученная модель и нужные файлы для предсказания, поэтому можно сразу выполнить:

```bash
python predict_author_profile.py --model-dir . --text "после пар решила дописать курсовую, кофе спасает"
```

Пример результата:

```text
Predicted class: female_18-24
Gender: female
Age group: 18-24
Confidence: 0.811
```

Точное значение уверенности может отличаться после переобучения модели.

## Повторное обучение модели

Для обучения модели на готовом датасете выполните:

```bash
python train_model.py --data telegram_author_profile_synthetic.csv --out-dir outputs --epochs 25
```

После обучения новые результаты будут сохранены в папку `outputs`:

```text
outputs/
├── author_profile_mlp.pt
├── tfidf_vectorizer.joblib
├── label_encoder.joblib
├── classification_report.txt
├── classification_report.json
├── confusion_matrix.csv
├── confusion_matrix.png
├── training_curve.png
└── training_history.csv
```

Проверить переобученную модель можно так:

```bash
python predict_author_profile.py --model-dir outputs --text "после пар решил доделать лабораторную работу"
```

## Генерация нового датасета

При необходимости можно создать новый синтетический датасет:

```bash
python generate_dataset.py --samples-per-class 250 --out data/telegram_author_profile_synthetic.csv
```

После этого модель можно обучить на новом файле:

```bash
python train_model.py --data data/telegram_author_profile_synthetic.csv --out-dir outputs --epochs 25
```

## Описание основных файлов

### `generate_dataset.py`

Создаёт синтетические Telegram-подобные тексты для разных возрастных групп и двух полов. В датасете формируется общий класс вида `female_18-24` или `male_35-44`.

### `train_model.py`

Загружает датасет, делит данные на обучающую и тестовую выборки, преобразует тексты в TF-IDF признаки, обучает MLP-модель и сохраняет результаты обучения.

### `predict_author_profile.py`

Загружает сохранённую модель, TF-IDF векторизатор и кодировщик классов, после чего определяет пол и возрастную группу по новому тексту.

### `create_report_assets.py`

Создаёт дополнительные изображения для пояснительной записки: распределение классов и схему конвейера обработки данных.

## Результаты работы

В проекте уже приложены результаты обучения:

- `classification_report.txt` — precision, recall, f1-score и accuracy;
- `confusion_matrix.png` — матрица ошибок;
- `training_curve.png` — график изменения ошибки на обучении и проверке;
- `training_history.csv` — история обучения по эпохам.

## Ограничения

Проект является учебным. Датасет синтетический, поэтому модель подходит для демонстрации методов машинного обучения, но не должна использоваться для реального профилирования пользователей без отдельной проверки на реальных размеченных данных.

Для работы с реальными Telegram-данными необходимы согласие пользователей, обезличивание, правовое основание обработки данных и проверка качества модели на независимой выборке.
