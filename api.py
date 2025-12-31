import asyncio
import uvicorn
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dask_analyse import get_info, get_count, get_avg, get_rate_devices, get_brand_info, save_training_data
from mdb_con import export_to_parquet
from model.model import model


async def background_analyze():
    """Асинхронная функция для обновления parquet файла"""
    while True:
        try:
            export_to_parquet()
            print("Обновлен файл для анализа.")
            save_training_data()
            print("Создан файл для обучения LLM на основе комментариев.")
        except Exception as e:
            print(f"Ошибка при обновлении: {e}")
        await asyncio.sleep(300)  # 3 минуты


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Запускаем фоновую задачу при старте приложения
    task = asyncio.create_task(background_analyze())
    yield
    task.cancel()

app = FastAPI(lifespan=lifespan)

# Middleware можно использовать для создания сайта
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def home():
    return {
        "http://localhost:8000/info": "Возвращает общую информацию о парсере",
        "http://localhost:8000/count": "Возвращает количество товаров по категории и количество устройств по брендам",
        "http://localhost:8000/rate-devices": "Возвращает рейтинг устройств",
        "http://localhost:8000/avg/rate": "Возвращает средний рейтинг по категории, бренду и году",
        "http://localhost:8000/avg/price": "Возвращает среднюю цену по категории, бренду и году",
        "http://localhost:8000/brand/{brand_name}": "Возвращает статистику по бренду",
        "http://localhost:8000/ai/?q={prompt}": "Запрос к обученной модели",
    }


@app.get("/{param}")
async def info(param: str):
    match param.lower():
        case "info":
            return get_info()
        case "count":
            return get_count()
        case "rate-devices":
            return get_rate_devices()
        case _:
            return {"Error 404": "Not found"}


@app.get("/avg/{avg_param}")
async def avg_rate(avg_param: str):
    return get_avg(param=avg_param.lower())


@app.get("/brand/{brand_name}")
async def brand_stats(brand_name: str):
    return get_brand_info(brand_name=brand_name.lower().capitalize())


@app.get("/ai/")
async def ai_assist(q: str = Query(..., description="Текстовый запрос к нейросети")):
    response = model.generate(prompt=q)
    return {"message": response}


def api_run():
    uvicorn.run(app, host="localhost", port=8000)


if __name__ == "__main__":
    api_run()

