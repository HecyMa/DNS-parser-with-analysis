"""
Команда для развертывания контейнера в докере (ОБЯЗАТЕЛЬНО):
docker run -d \
  --name mongodb \
  -p 27017:27017 \
  -e MONGO_INITDB_ROOT_USERNAME=admin \
  -e MONGO_INITDB_ROOT_PASSWORD=secret \
  mongo:latest
После этого запускайте скрипт main.py
"""

import os
import json
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import pyarrow as pa
import pyarrow.parquet as pq


MONGO_HOST = "localhost"
MONGO_PORT = 27017
MONGO_USERNAME = "admin"
MONGO_PASSWORD = "secret"


def create_connection():
    """Создает подключение к MongoDB и возвращает коллекцию"""
    global MONGO_HOST, MONGO_PORT, MONGO_USERNAME, MONGO_PASSWORD
    connection_string = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/?authSource=admin"
    try:
        client = MongoClient(connection_string, serverSelectionTimeoutMS=2000)
        client.admin.command('ping')
    except ConnectionFailure as e:
        print(f"Ошибка подключения: {e}")
        exit(1)
    db = client["tech_analytics"]
    collection = db["products"]

    return collection


def insert_data(document):
    """Вставляет значения"""
    collection = create_connection()
    collection.insert_one(document)
    print('Документ вставлен успешно.')


def show_data():
    """Просмотр вставленных документов"""
    collection = create_connection()
    documents = collection.find()
    for doc in documents:
        print(doc)
    print(f'Количество документов: {collection.count_documents({})}')


def clear_data():
    """Полностью очистить бд"""
    collection = create_connection()
    collection.delete_many({})
    print("Документы успешно удалены.")


def check_data(url: str):
    """
    Проверяет, есть ли товар в бд по url
    Если продукт есть в бд: return False
    Если продукта нет в бд: return True
    """
    collection = create_connection()
    exists = collection.find_one({"Ссылка": url.strip()})
    if exists:
        return False
    else:
        return True


def export_data(filename="products.json"):
    """Записывает все документы в JSON файл без _id"""
    collection = create_connection()
    documents = []
    for doc in collection.find({}):
        doc.pop('_id', None)
        documents.append(doc)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=4)
    print(f"Сохранено {len(documents)} документов в '{filename}'")


def export_to_parquet(output_path="./data/products.parquet"):
    """Экспорт данных из MongoDB в Parquet-файл, только если документов стало больше"""
    collection = create_connection()
    mongo_count = collection.count_documents({})
    # Если в MongoDB нет документов — выходим
    if mongo_count == 0:
        print("Нет документов для анализа.")
        return
    # Проверяем, существует ли уже Parquet-файл
    if os.path.exists(output_path):
        try:
            existing_table = pq.read_table(output_path)
            existing_count = existing_table.num_rows
        except Exception as e:
            print(f"Не удалось прочитать существующий файл: {e}")
            existing_count = 0
    else:
        existing_count = 0
    # Сохраняем только если документов стало больше
    if mongo_count <= existing_count:
        return

    documents = []
    for doc in collection.find({}, {"_id": 0}):
        documents.append(doc)
    print(f"Найдено {len(documents)} документов. Сохранение в Parquet.")
    table = pa.Table.from_pylist(documents)
    pq.write_table(table, output_path, compression='snappy')
