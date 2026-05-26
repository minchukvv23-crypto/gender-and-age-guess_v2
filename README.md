# Курсовой проект: предсказание пола и возраста автора Telegram-текста

Проект содержит синтетический датасет, код генерации данных, код обучения нейронной сети и пример использования модели.

## Структура

- `data/telegram_author_profile_synthetic.csv` — синтетический датасет.
- `src/generate_dataset.py` — генерация датасета.
- `src/train_model.py` — обучение нейронной сети MLP на TF-IDF признаках.
- `src/predict_author_profile.py` — предсказание класса для нового текста.
- `outputs/` — метрики, графики, веса модели после обучения.

## Установка

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

## Генерация датасета

```bash
python src/generate_dataset.py --samples-per-class 250 --out data/telegram_author_profile_synthetic.csv
```

## Обучение

```bash
python src/train_model.py --data data/telegram_author_profile_synthetic.csv --out-dir outputs --epochs 25
```

## Проверка на новом тексте

```bash
python src/predict_author_profile.py --model-dir outputs --text "после пар решила дописать курсовую, кофе спасает"
```

## Важное ограничение

Датасет синтетический и подходит для учебной демонстрации. Для реальной работы с Telegram-данными нужны согласие пользователей, обезличивание, правовое основание обработки и отдельная проверка модели на реальных размеченных данных.
