import requests
from datetime import datetime, timedelta
import os
import random
from concurrent.futures import ThreadPoolExecutor

"""General settings"""
QUALITY = 1024 #4096
NUM_THREADS = 32  


"""Date choosing"""
NUM_DAYS = 420
START_YEAR = 2011
hours = list(range(24)) 
minutes = list(range(61)) 
seconds = [38,10,25,41,54,39]
random_dates = [generate_random_date() for _ in range(NUM_DAYS)]

# Funkcja generująca losową datę
def generate_random_date(start_year=START_YEAR, end_date=datetime.today()):
    start_date = datetime(start_year, 1, 1)
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    random_date = start_date + timedelta(days=random_days)
    return random_date.strftime("%Y-%m-%d")
# Funkcja generująca URL obrazu na podstawie daty, godziny i minut
def generate_image_url(date_str, hour, minute, second,quality=QUALITY):
    # Data w formacie YYYYMMDD (np. 20250210)
    date_part1= date_str.replace("-", "/")
    date_part2 = date_str.replace("-", "")
    # Godzina i minuta (np. 114038 -> godzina 11, minuta 40)
    time_part = f"{hour:02}{minute:02}{second:02}" 
    # Generowanie pełnego URL
    url = f"https://sdo.gsfc.nasa.gov/assets/img/browse/{date_part1}/{date_part2}_{time_part}_{quality}_HMII.jpg"    
    return url
def download_image(session, url, img_name):
    try:
        response = session.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            os.makedirs("sdo_data", exist_ok=True)  # Tworzy folder, jeśli nie istnieje
            img_path = os.path.join("sdo_data", img_name)
            with open(img_path, "wb") as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            print(f"Pobrano: {url}")
            return True
    except requests.RequestException as e:
        print(f"Błąd pobierania {img_name}: {e}")
    return False




def minutes_loop(session, date, hour, minutes, seconds):
    for minute in minutes:
        for second in seconds:
            url = generate_image_url(date, hour, minute, second)
            img_name = f"{date}_{hour}_{minute}_{second}.jpg"
            if download_image(session, url, img_name):
                return
def hours_loop(session, date, hours, minutes, seconds):
    for hour in hours:
        print(f"Data: {date} Godzina: {hour}")
        minutes_loop(session, date, hour, minutes, seconds)





# Ustawienia

with requests.Session() as session:
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        tasks = []
        for date in random_dates:
            tasks.append(executor.submit(hours_loop, session, date, hours, minutes, seconds))
        # Czekamy na zakończenie wszystkich wątków
        for task in tasks:
            task.result()  # Zbieramy wyniki zakończonych zadań


            