import sys
import re
import random

from bs4 import BeautifulSoup
from time import sleep
from tqdm import tqdm
from mdb_con import insert_data, check_data
from typing import Optional, Tuple

import undetected_chromedriver as uc
from undetected_chromedriver import Chrome

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def is_valid(record: dict) -> bool:
    """
    Проверяет валидность словаря
    Если все ключи есть в словаре, то выводит True, в противном случае - False.
    """
    required_keys = {
        'Категория', 'Наименование', 'Цена', 'Рейтинг',
        'Ссылка', 'Описание', 'Характеристики', 'Отзывы', 'Всего_отзывов'
    }

    return required_keys.issubset(record.keys())


def get_max_pages(driver: Chrome, url: str) -> Optional[int]:
    """Возвращает максимальное число страниц в категории"""
    try:
        driver.get(url)
        sleep(1)
        driver.execute_script("window.stop();")
        max_page = 1
        soup = BeautifulSoup(driver.page_source, 'lxml')

        # Поиск всех виджетов для перемещения по странице
        all_pages = soup.find_all('li', class_='pagination-widget__page')
        if len(all_pages) >= 2:
            max_page = max(
                int(item.get('data-page-number', 0))
                for item in all_pages
                if item.get('data-page-number') and item.get('data-page-number').isdigit()
            )

        return max_page

    except Exception as error:
        print(f"Ошибка получения количества страниц: {error}")


def get_product_urls_from_page(driver: Chrome, url: str) -> Tuple[list[str], list[str]]:
    """Получает все ссылки на продукты и возвращает в функцию get_all_product_urls_in_category"""
    try:
        # Кидаем ссылку на страницу парсеру
        driver.get(url)
        sleep(1)
        driver.execute_script("window.stop();")
        urls_char = []
        urls_opin = []
        soup = BeautifulSoup(driver.page_source, 'lxml')
        # Проходимся по странице, забираем все ссылки, преобразуем их и добавляем в список
        for link in soup.select('a.catalog-product__name.ui-link.ui-link_black'):
            href = link.get('href')
            if href:
                base_url = "https://www.dns-shop.ru" + href.split('?')[0].rstrip('/')
                urls_char.append(base_url + "/characteristics/")
                urls_opin.append(base_url + "/opinion/")
        return urls_char, urls_opin
    except Exception as e:
        print("Ошибка при сборе ссылок:", e)
        return [], []


def get_all_product_urls_in_category(driver: Chrome, url: str) -> Tuple[list[str], list[str]]:
    """Получает все ссылки на продукты и возвращает в функцию main"""
    all_char_urls = []
    all_opin_urls = []

    urls_char, urls_opin = get_product_urls_from_page(driver, url)

    all_char_urls.extend(urls_char)
    all_opin_urls.extend(urls_opin)
    sleep(random.uniform(0.3, 0.8))

    return all_char_urls, all_opin_urls


def parse_characteristics_page(driver: Chrome, url: str) -> dict:
    """Парсит характеристики товара"""
    try:
        driver.get(url)
        sleep(1)
        driver.execute_script("window.stop();")
        soup = BeautifulSoup(driver.page_source, 'lxml')
        name = soup.find('h1', class_="title")

        price_div = soup.find('div', class_='product-buy__price')
        price_number = 0

        # Удаляем вложенный span
        if price_div:
            prev_span = price_div.find('span', class_='product-buy__prev')
            if prev_span:
                prev_span.decompose()  # Удаляет тег из дерева

            # Получаем текущую цену
            current_price_text = price_div.get_text(strip=True)
            cleaned = re.sub(r'[^\d\s]', '', current_price_text).replace('\xa0', ' ')
            price_number = int(cleaned.replace(' ', ''))

        rate = soup.find('a', class_="header-product__link_rating")
        desc = soup.find('div', class_="product-card-description-text")

        category = "Не указана"
        for span in soup.find_all('span'):
            if span.get('data-go-back-catalog') is not None:
                category = span.get_text(strip=True).lstrip(': ')
                break

        charcs = soup.find_all('div', class_="product-characteristics__spec-title")
        cvalue = soup.find_all('div', class_="product-characteristics__spec-value")
        tech_spec = {
            title.get_text(strip=True): value.get_text(strip=True)
            for title, value in zip(charcs, cvalue)
        }

        # Очищаем название от слова "Характеристики"
        raw_name = name.get_text(strip=True) if name else "Не указано"
        clean_name = raw_name.replace("Характеристики", "").strip()

        return {
            "Категория": category,
            "Наименование": clean_name,
            "Цена": price_number,
            "Рейтинг": rate.get_text(strip=True) if rate else "Нет рейтинга",
            "Ссылка": url.replace("/characteristics/", "/"),
            "Описание": desc.get_text(strip=True) if desc else "Описание отсутствует",
            "Характеристики": tech_spec,
        }

    except Exception as error:
        return {"Ссылка": url, "Ошибка_характеристики": str(error)}


def parse_opinion_page(driver: Chrome, url: str) -> dict:
    """Парсит страницу с отзывами на товар"""
    try:
        driver.get(url)
        sleep(1)
        driver.execute_script("window.stop();")

        wait = WebDriverWait(driver, 10)
        button = wait.until(
            EC.element_to_be_clickable((By.XPATH,
            '//div[contains(@class, "ow-filters__count-filter-btn") and contains(text(), "Только к этой модели")]'))
        )
        button.click()
        sleep(2)
        reviews = []
        soup_n = BeautifulSoup(driver.page_source, 'lxml')
        review_counts = soup_n.find_all('div', class_='ow-filters__count-filter-btn')[-1].get_text().split()[-1]
        for _ in range((int(review_counts) - 4) // 10 + 1):
            try:
                show_more_button = wait.until(
                    EC.element_to_be_clickable(
                        (By.XPATH, "//button[contains(@class, 'dget__more') and contains(text(), 'Показать ещё')]"))
                )
                driver.execute_script("arguments[0].scrollIntoView({block: 'center', behavior: 'smooth'});",
                                      show_more_button)
                show_more_button.click()
                sleep(1.5)
                soup_n = BeautifulSoup(driver.page_source, 'lxml')

            except Exception as error:
                print(f"Не удалось нажать 'Показать ещё': {error}")
                break

        opinion_blocks = soup_n.select("div.ow-opinion[data-role='opinion']")
        if not opinion_blocks:
            return {"Отзывы": [], "Всего_отзывов": 0}

        for block in opinion_blocks:
            try:
                # Автор
                author_elem = block.select_one(".profile-info__name")
                author = author_elem.get_text(strip=True) if author_elem else "Аноним"

                # Дата
                date_elem = block.select_one(".ow-opinion__date")
                date = date_elem.get_text(strip=True) if date_elem else ""

                # Реальный покупатель
                is_real = bool(block.select_one("[data-real-buyer]"))

                # Общий рейтинг (звезды)
                stars = len(block.select(".star-rating__star[data-state='selected']"))

                # Оценки по категориям
                category_ratings = {}
                rating_items = block.select(".opinion-rating-slider__tab")
                for item in rating_items[1:]:
                    num_elem = item.select_one("span:first-child")
                    name_elem = item.select_one(".opinion-rating-slider__tab-title_name")
                    if num_elem and name_elem:
                        name = name_elem.get_text(strip=True).rstrip(":")
                        try:
                            value = int(num_elem.get_text(strip=True))
                            category_ratings[name] = value
                        except (ValueError, AttributeError):
                            continue

                # Срок использования
                usage_period = ""
                usage_elem = block.select_one(".ow-opinion__info-desc")
                if usage_elem:
                    usage_period = usage_elem.get_text(strip=True)

                # Достоинства / Недостатки / Комментарий
                texts = {}
                for text_block in block.select(".ow-opinion__text"):
                    title_elem = text_block.select_one(".ow-opinion__text-title")
                    desc_elem = text_block.select_one(".ow-opinion__text-desc")
                    if title_elem and desc_elem:
                        title = title_elem.get_text(strip=True)
                        desc = desc_elem.get_text(strip=True)
                        texts[title] = desc

                reviews.append({
                    "Автор": author,
                    "Дата": date,
                    "Реальный покупатель": is_real,
                    "Общий рейтинг": stars,
                    "Оценки по категориям": category_ratings,
                    "Срок использования": usage_period,
                    "Достоинства": texts.get("Достоинства", ""),
                    "Недостатки": texts.get("Недостатки", ""),
                    "Комментарий": texts.get("Комментарий", ""),
                })
            except Exception as error:
                print(f"Ошибка при парсинге одного отзыва: {error}")
                continue

        return {
            "Отзывы": reviews,
            "Всего_отзывов": int(review_counts)
        }

    except:
        return {"Отзывы": [], "Всего_отзывов": 0}


def main(driver: Chrome, url: str):
    """Функция для создания парсера"""
    driver = driver
    driver.execute_script("window.stop();")

    try:
        all_char_urls = []
        all_opin_urls = []
        char_urls, opin_urls = get_all_product_urls_in_category(driver, url)
        all_char_urls.extend(char_urls)
        all_opin_urls.extend(opin_urls)
        if not all_char_urls:
            print('\nТоваров не найдено')
        for char_url, opin_url in tqdm(zip(all_char_urls, all_opin_urls), total=len(all_char_urls), ncols=70, unit='товар', colour='blue', file=sys.stdout):
            if check_data(char_url.replace("/characteristics/", "/")) and check_data(url.replace("/opinion/", "/")):
                char_data = parse_characteristics_page(driver, char_url)
                opin_data = parse_opinion_page(driver, opin_url)
                if char_data is not None and opin_data is not None:
                    result = {**char_data, **opin_data}
                    if is_valid(result):
                        insert_data(result)

    except Exception as error:
        print(f"Ошибка парсинга: {error}")


def run():
    """Основная функция для запуска"""
    options = uc.ChromeOptions()
    options.page_load_strategy = 'eager'
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-images")
    options.add_argument("--disable-plugins")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-gpu")
    options.add_argument("--blink-settings=imagesEnabled=false")
    options.add_argument("--log-level=3")
    options.add_argument("--silent")
    options.add_argument("--disable-logging")
    options.add_argument("--disable-background-timer-throttling")
    options.add_argument("--disable-backgrounding-occluded-windows")
    options.add_argument("--disable-renderer-backgrounding")
    driver = uc.Chrome(options=options)
    driver.set_window_size(1200, 900)

    urls = [
        'https://www.dns-shop.ru/catalog/17a8a01d16404e77/smartfony/?p={page}',
        'https://www.dns-shop.ru/catalog/17a8a05316404e77/planshety/?p={page}',
        'https://www.dns-shop.ru/catalog/251c82c88ed24e77/smart-chasy-i-braslety/?p={page}',
        'https://www.dns-shop.ru/catalog/17a9ef1716404e77/naushniki-i-garnitury/?p={page}',
        'https://www.dns-shop.ru/catalog/17a8ae4916404e77/televizory/?p={page}',
        'https://www.dns-shop.ru/catalog/17a8c51716404e77/saundbary/?p={page}',
        'https://www.dns-shop.ru/catalog/d79905f0113ab6df/vertikalnye-i-ruchnye-pylesosy/?p={page}',
        'https://www.dns-shop.ru/catalog/c01df46f39137fd7/stiralnye-mashiny/?p={page}',
        'https://www.dns-shop.ru/catalog/4e2a7cdb390b7fd7/holodilniki/?p={page}',
        'https://www.dns-shop.ru/catalog/17a8c89d16404e77/mikrovolnovye-pechi/?p={page}',
        'https://www.dns-shop.ru/catalog/17a8d3a316404e77/kondicionery/?p={page}',
        'https://www.dns-shop.ru/catalog/46215ff3b2cb7fd7/vneshnie-ssd-nakopiteli/?p={page}',
        'https://www.dns-shop.ru/catalog/17a892f816404e77/noutbuki/?p={page}',
        'https://www.dns-shop.ru/catalog/17a8d15716404e77/vstraivaemye-mikrovolnovye-pechi/?p={page}',
        'https://www.dns-shop.ru/catalog/17a8d1c216404e77/vstraivaemye-posudomoechnye-mashiny/?p={page}',
        'https://www.dns-shop.ru/catalog/17a8d26216404e77/vstraivaemye-holodilniki/?p={page}',
        'https://www.dns-shop.ru/catalog/17a8d0b816404e77/varochnye-paneli-elektricheskie/?p={page}',
        'https://www.dns-shop.ru/catalog/17a9fce216404e77/varochnye-paneli-gazovye/?p={page}',
        'https://www.dns-shop.ru/catalog/17a8d18c16404e77/duhovye-shkafy-elektricheskie/?p={page}',
        'https://www.dns-shop.ru/catalog/17a9e6e016404e77/vytyazhki/?p={page}',
        'https://www.dns-shop.ru/catalog/17a8a26516404e77/kabeli-dlya-mobilnyh-ustrojstv/?p={page}',
        'https://www.dns-shop.ru/catalog/17a8a30616404e77/setevye-zaryadnye-ustrojstva/?p={page}',
        'https://www.dns-shop.ru/catalog/2b911f3c621a36eb/servernye-ssd-m2/?p={page}',
        'https://www.dns-shop.ru/catalog/1023687c7ba7a69d/servernye-ssd/?p={page}',
        'https://www.dns-shop.ru/catalog/17a89a3916404e77/operativnaya-pamyat-dimm/?p={page}',
        'https://www.dns-shop.ru/catalog/17a9b91b16404e77/operativnaya-pamyat-so-dimm/?p={page}',
        'https://www.dns-shop.ru/catalog/17a8e3e116404e77/proektory/?p={page}'
    ]

    for link in urls:
        sleep(1)
        driver.execute_script("window.stop();")
        max_pages = get_max_pages(driver, link.format(page=1))
        for page in range(1, max_pages + 1):
            main(driver, link.format(page=page))

    driver.quit()


if __name__ == '__main__':
    run()
