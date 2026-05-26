"""Generate a synthetic Telegram-style dataset for author profiling.

The dataset is educational and does not contain real Telegram posts, usernames,
phone numbers, IDs, or other personal data.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


RANDOM_SEED = 42

GENDER_FORMS = {
    "male": {
        "ru": "мужской",
        "verbs": [
            "решил", "написал", "сходил", "купил", "начал", "устал", "посмотрел",
            "обновил", "забыл", "вернулся", "заказал", "поставил", "подумал", "проверил",
            "разобрался", "планировал", "выбрал", "готов"
        ],
        "self": ["я сам", "мне кажется", "я сегодня", "в итоге я", "короче я"],
    },
    "female": {
        "ru": "женский",
        "verbs": [
            "решила", "написала", "сходила", "купила", "начала", "устала", "посмотрела",
            "обновила", "забыла", "вернулась", "заказала", "поставила", "подумала", "проверила",
            "разобралась", "планировала", "выбрала", "готова"
        ],
        "self": ["я сама", "мне кажется", "я сегодня", "в итоге я", "короче я"],
    },
}

AGE_GROUPS = {
    "14-17": {
        "topics": [
            "контрольная по алгебре", "домашка по истории", "подготовка к ЦЭ", "репетитор по английскому",
            "олимпиада в школе", "урок информатики", "классный чат", "родители снова спрашивают про оценки",
            "пробник по математике", "школьный проект"
        ],
        "objects": ["тетрадь", "учебник", "расписание", "дневник", "школьный рюкзак", "презентацию"],
        "time": ["после школы", "на перемене", "к вечеру", "перед контрольной", "с утра"],
        "style": ["лол", "кто тоже так?", "жесть", "ахах", "надо выжить эту неделю"],
    },
    "18-24": {
        "topics": [
            "сессия в универе", "пары с утра", "лаба по программированию", "курсовая работа", "зачётная неделя",
            "общага шумит", "стажировка", "дедлайн по проекту", "экзамен", "лекция онлайн"
        ],
        "objects": ["конспект", "ноутбук", "билет к экзамену", "код", "отчёт", "методичку"],
        "time": ["после пар", "ночью", "перед зачётом", "между парами", "к дедлайну"],
        "style": ["ну такое", "надо закрыть хвосты", "кофе спасает", "кто в теме?", "минус сон"],
    },
    "25-34": {
        "topics": [
            "рабочий проект", "созвон с командой", "планирование спринта", "собеседование", "ипотека",
            "тренировка после работы", "отпуск", "новый сервис", "дедлайн у клиента", "офисный день"
        ],
        "objects": ["план задач", "таблицу", "презентацию", "резюме", "отчёт", "календарь"],
        "time": ["после работы", "до обеда", "к пятнице", "вечером", "с утра"],
        "style": ["надо не забыть", "выгорание рядом", "зато продуктивно", "всё по плану", "без паники"],
    },
    "35-44": {
        "topics": [
            "ремонт на кухне", "родительское собрание", "школа у ребёнка", "рабочее совещание", "семейный бюджет",
            "запись к врачу", "поездка за город", "покупки для дома", "проверка уроков", "отпуск с семьёй"
        ],
        "objects": ["смету", "список покупок", "документы", "расписание ребёнка", "платёж", "план на неделю"],
        "time": ["после работы", "на выходных", "к вечеру", "с утра", "перед ужином"],
        "style": ["дел хватает", "главное всё успеть", "будни как будни", "семья ждёт", "без расписания никак"],
    },
    "45+": {
        "topics": [
            "дачные дела", "квитанции за квартиру", "новости района", "запись в поликлинику", "встреча с родственниками",
            "рынок утром", "ремонт старого телефона", "садовые работы", "внуки приехали", "поездка на дачу"
        ],
        "objects": ["квитанцию", "список лекарств", "пакет с рынка", "рассаду", "старые фотографии", "план поездки"],
        "time": ["с утра", "после обеда", "на выходных", "к вечеру", "вчера"],
        "style": ["надо спокойно разобраться", "времени всё равно мало", "главное здоровье", "как-то так", "дела житейские"],
    },
}

COMMON_OPENERS = [
    "в телеге опять обсуждают, что", "сегодня в чате написали, что", "маленькое наблюдение:",
    "пишу сюда, чтобы не забыть:", "коммент к посту:", "у кого так было?", "коротко про день:"
]

COMMON_ENDINGS = [
    "🙂", "без лишних слов", "посмотрим, что получится", "потом отпишусь", "в общем такие дела",
    "если что, обновлю пост", "надеюсь, завтра будет проще", "держу в голове"
]

TEMPLATES = [
    "{opener} {time} {verb} {obj}, потому что {topic}; {style}.",
    "{time} {self_phrase} {verb} про {topic}. {style}, {ending}.",
    "{opener} {topic}. Я {verb} {obj} и теперь думаю, что делать дальше. {ending}.",
    "{topic}: {self_phrase} {verb} всё проверить {time}. {style}.",
    "Не ожидал(а), что {topic} так выбьет из режима. {time} {verb} {obj}, {ending}.",
    "В комментариях спорят про {topic}, а я {verb} {obj}. {style}.",
    "{self_phrase} {verb} маленький апдейт: {topic}, {time}, {ending}.",
]

NEUTRAL_NOISE = [
    "канал снова прислал уведомление в самый неподходящий момент",
    "интернет сегодня работает странно",
    "в городе резко поменялась погода",
    "лента стала какой-то слишком шумной",
    "в комментариях как всегда много разных мнений",
]


def make_text(gender: str, age_group: str) -> str:
    g = GENDER_FORMS[gender]
    a = AGE_GROUPS[age_group]
    template = random.choice(TEMPLATES)
    text = template.format(
        opener=random.choice(COMMON_OPENERS),
        time=random.choice(a["time"]),
        verb=random.choice(g["verbs"]),
        obj=random.choice(a["objects"]),
        topic=random.choice(a["topics"]),
        style=random.choice(a["style"]),
        ending=random.choice(COMMON_ENDINGS),
        self_phrase=random.choice(g["self"]),
    )
    if random.random() < 0.28:
        text += " " + random.choice(NEUTRAL_NOISE) + "."
    if random.random() < 0.12:
        text = text.lower()
    return " ".join(text.split())


def generate_dataset(samples_per_class: int) -> pd.DataFrame:
    random.seed(RANDOM_SEED)
    rows = []
    for gender in GENDER_FORMS:
        for age_group in AGE_GROUPS:
            for _ in range(samples_per_class):
                rows.append(
                    {
                        "text": make_text(gender, age_group),
                        "gender": gender,
                        "gender_ru": GENDER_FORMS[gender]["ru"],
                        "age_group": age_group,
                        "class_label": f"{gender}_{age_group}",
                        "source_type": random.choice(["post", "comment"]),
                    }
                )
    random.shuffle(rows)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-class", type=int, default=250)
    parser.add_argument("--out", type=Path, default=Path("data/telegram_author_profile_synthetic.csv"))
    args = parser.parse_args()
    df = generate_dataset(args.samples_per_class)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} rows to {args.out}")
    print(df["class_label"].value_counts().sort_index())


if __name__ == "__main__":
    main()
