import re
import json
from typing import Optional, Any, Dict, List
from datetime import datetime
from functools import lru_cache
import numpy
import pandas as pd
from pandas import DataFrame, Series
import dask.dataframe as dd


def extract_base_model(name: str) -> str:
    """Извлекает базовую модель из названия, убирая спецификации."""
    if not isinstance(name, str):
        return "Неизвестно"

    name = re.sub(r'\([^)]*\)', '', name)
    words = name.strip().split()
    base = []
    spec_patterns = ['/', 'гб', 'gb', 'ram', 'rom', 'мб', 'mb', 'tb', 'тб', 'ghz', 'ггц']

    for w in words:
        if any(spec in w.lower() for spec in spec_patterns):
            break
        if w.isdigit() and base:
            break
        base.append(w)

    return " ".join(base) if base else name.strip()


def extract_brand_from_model_or_name(row: Series) -> str:
    """Извлекает бренд как первое слово из 'Модель' в характеристиках или 'Наименование'."""
    char = row.get("Характеристики", {})

    if isinstance(char, dict):
        model = char.get("Модель")
        if isinstance(model, str) and model.strip():
            return model.strip().split()[0]

    name = row.get("Наименование", "")
    if isinstance(name, str) and name.strip():
        return name.split()[0]

    return "Неизвестно"


def extract_year(char: dict) -> Optional[int]:
    """Извлекает год релиза."""
    val = char.get("Год релиза", "")
    return int(val) if isinstance(val, str) and val.isdigit() else None


def normalize_char(x: Any) -> Dict[str, Any]:
    """Приводит характеристики к dict даже при None/str/bool."""
    return x if isinstance(x, dict) else {}


@lru_cache(maxsize=2)
def load_and_process_data(parquet_path: str = "./data/products.parquet") -> DataFrame:
    """
    Загружает и обрабатывает данные из parquet файла.
    Использует кэширование для повторных вызовов.
    """
    # Загрузка через Dask для больших файлов
    ddf = dd.read_parquet(parquet_path)
    pdf = ddf.compute()

    # Базовые преобразования
    pdf["Характеристики"] = pdf["Характеристики"].apply(normalize_char)
    pdf["Рейтинг"] = pd.to_numeric(pdf["Рейтинг"], errors="coerce")
    pdf["Всего_отзывов"] = pd.to_numeric(pdf["Всего_отзывов"], errors="coerce")
    pdf["Цена"] = pd.to_numeric(pdf["Цена"], errors="coerce")

    # Извлечение дополнительных полей
    pdf["Базовая_модель"] = pdf["Наименование"].apply(extract_base_model)
    pdf["Бренд"] = pdf.apply(extract_brand_from_model_or_name, axis=1)
    pdf["Год"] = pdf["Характеристики"].apply(extract_year)

    return pdf


def get_info(parquet_path: str = "./data/products.parquet") -> dict:
    """Возвращает общую информацию о данных."""
    pdf = load_and_process_data(parquet_path)

    return {
        "total_products": len(pdf),
        "total_reviews": int(pdf["Всего_отзывов"].fillna(0).sum()),
        "last_parsing": datetime.now().isoformat()
    }


def get_count(parquet_path: str = "./data/products.parquet") -> dict:
    """Возвращает количество товаров по категории и брендам."""
    pdf = load_and_process_data(parquet_path)

    return {
        "category_distribution": pdf["Категория"].value_counts().to_dict(),
        "brand_distribution": pdf["Бренд"].value_counts().to_dict()
    }


def get_avg(param: str, parquet_path: str = "./data/products.parquet") -> dict:
    """Возвращает средние рейтинг или цену по категории, бренду и году."""
    pdf = load_and_process_data(parquet_path)

    if param == "rate":
        agg_data = pdf.dropna(subset=["Рейтинг"])
        col = "Рейтинг"
    elif param == "price":
        agg_data = pdf.dropna(subset=["Цена"])
        col = "Цена"
    else:
        return {"error": "Invalid parameter. Use 'rate' or 'price'"}

    # Группировки
    result = {
        f"avg_{param}_by_category": agg_data.groupby("Категория")[col].mean().round(2).to_dict(),
        f"avg_{param}_by_brand": agg_data.groupby("Бренд")[col]
        .mean()
        .sort_values(ascending=False)
        .head(20)
        .round(2)
        .to_dict(),
    }

    # Добавляем группировку по году если есть данные
    if pdf["Год"].notna().any():
        yearly_agg = pdf.dropna(subset=["Год", col]).groupby("Год")[col].mean().round(2).sort_index()
        result[f"avg_{param}_by_year"] = yearly_agg.to_dict()

    return result


def get_rate_devices(parquet_path: str = "./data/products.parquet") -> dict:
    """Возвращает рейтинг устройств, отсортированный от высшего к низшему."""
    pdf = load_and_process_data(parquet_path)

    # Фильтрация и сортировка
    rated = pdf.dropna(subset=["Рейтинг"]).sort_values("Рейтинг", ascending=False)

    # Формирование результата
    result_list = rated[["Наименование", "Рейтинг", "Всего_отзывов"]] \
        .assign(
        Рейтинг=lambda x: x["Рейтинг"].round(2),
        Всего_отзывов=lambda x: x["Всего_отзывов"].fillna(0).astype(int)
    ) \
        .to_dict(orient="records")

    return {"devices_by_rating": result_list}


def get_brand_info(brand_name: str, parquet_path: str = "./data/products.parquet") -> dict:
    """Возвращает статистику по запрашиваемому бренду."""
    pdf = load_and_process_data(parquet_path)

    # Фильтрация бренда
    brand_mask = pdf["Бренд"] == brand_name
    brand_df = pdf[brand_mask].copy()

    if brand_df.empty:
        return {"error": f"Brand '{brand_name}' not found"}

    # Базовая статистика
    device_count = len(brand_df)
    total_devices = len(pdf)
    share_percent = (device_count / total_devices * 100) if total_devices > 0 else 0

    # Числовая статистика
    stats = {
        'brand': brand_name,
        "device_count": device_count,
        "avg_rating": round(brand_df["Рейтинг"].mean(), 2),
        "avg_price": round(brand_df["Цена"].mean(), 2),
        "min_price": brand_df["Цена"].min(),
        "max_price": brand_df["Цена"].max(),
        "share_percent": round(share_percent, 2),
        "total_reviews": int(brand_df["Всего_отзывов"].sum())
    }

    # Список устройств
    stats["devices"] = brand_df[["Наименование", "Рейтинг", "Всего_отзывов", "Цена"]] \
        .assign(
        Рейтинг=lambda x: x["Рейтинг"].round(2),
        Всего_отзывов=lambda x: x["Всего_отзывов"].fillna(0).astype(int),
        Цена=lambda x: x["Цена"].round(2)
    ) \
        .to_dict(orient="records")

    return stats


def clean_text(text: str) -> str:
    """Функция для очистки текста"""
    if not text:
        return ""

    # Удаляем эмодзи (всё, что не ASCII и не кириллица/латиница/цифры/пробелы/пунктуация)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002500-\U00002BEF"
        "\U00002702-\U000027B0"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"
        "\U0001FA70-\U0001FAFF"
        "\U0001F004-\U0001F0CF"
        "\U0001F170-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub(r'', text)

    # Удаляем повторяющиеся знаки препинания и символы (>2 подряд)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # Удаляем лишние пробелы и переносы
    text = re.sub(r'\s+', ' ', text).strip()

    # Убираем лишние точки и точки с запятой в конце
    text = re.sub(r'[.;,\s]+$', '', text)

    return text.strip()


def extract_reviews_data(parquet_path: str = "./data/products.parquet") -> List[Dict]:
    """Функция для извлечения данных о комментариях на товары для обучения LLM модели"""
    ddf = dd.read_parquet(parquet_path)
    pdf = ddf.compute()

    training_examples = []

    for _, row in pdf.iterrows():
        product_name = row.get("Наименование", "Неизвестный товар")
        reviews = row.get("Отзывы", [])

        if isinstance(reviews, numpy.ndarray):
            reviews = reviews.tolist()
        if not isinstance(reviews, list):
            continue

        for review in reviews:
            if not isinstance(review, dict):
                continue

            pros = clean_text(str(review.get("Достоинства", "")).strip())
            cons = clean_text(str(review.get("Недостатки", "")).strip())
            comment = clean_text(str(review.get("Комментарий", "")).strip())

            # Пропускаем пустые отзывы
            if not (pros or cons or comment):
                continue

            parts = []
            if pros:
                parts.append(f"Достоинства: {pros}")
            if cons:
                parts.append(f"Недостатки: {cons}")
            if comment:
                parts.append(f"Комментарий: {comment}")

            response = ". ".join(parts) + "."

            training_examples.append({
                "messages": [
                    {"role": "user", "content": f"Опиши достоинства и недостатки товара: {product_name}"},
                    {"role": "assistant", "content": response}
                ]
            })

    return training_examples


def save_training_data(path: str = "./model/data/reviews_train.jsonl"):
    """Сохранение данных в .jsonl файле для обучения модели"""
    data = extract_reviews_data()
    with open(path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
