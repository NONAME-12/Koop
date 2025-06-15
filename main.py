from PyQt5 import QtWidgets, QtCore, QtGui
import sys
from PyQt5 import QtWidgets
import pyautogui
import threading
from pynput import keyboard
import pytesseract
from PIL import Image
import base64
import requests
import time
import numpy as np
import re
import cv2
from difflib import SequenceMatcher
import json
from pynput import mouse
import subprocess
import os

class ApiKeyDialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Введите API-ключ OpenAI')
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)
        self.api_key = None
        self.init_ui()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()
        self.label = QtWidgets.QLabel('Пожалуйста, введите ваш OpenAI API-ключ:')
        self.input = QtWidgets.QLineEdit()
        self.button = QtWidgets.QPushButton('OK')
        self.button.clicked.connect(self.accept_key)
        layout.addWidget(self.label)
        layout.addWidget(self.input)
        layout.addWidget(self.button)
        self.setLayout(layout)

    def accept_key(self):
        key = self.input.text().strip()
        if key:
            self.api_key = key
            self.accept()
        else:
            QtWidgets.QMessageBox.warning(self, 'Ошибка', 'API-ключ не может быть пустым!')

stop_flag = threading.Event()
api_key = None

def send_image_to_gpt4o_answer_only(api_key, image_path):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    with open(image_path, "rb") as img_file:
        img_data = img_file.read()
    img_b64 = base64.b64encode(img_data).decode()
    prompt = (
        "На изображении находится вопрос теста и варианты ответов. "
        "Если в вариантах ответа есть буквенная нумерация (a., b., c. и т.д.), то отвечай только буквой (например, 'b.'). "
        "Если буквенной нумерации нет, то отвечай точным текстом правильного варианта. Это ОБЯЗАТЕЛЬНОЕ ПРАВИЛО! Если в вариантах НЕТ буквенной нумерации, то отвечай точным текстом правильного варианта."
        "Найди правильный вариант ответа, который В ТОЧНОСТИ совпадает с вариантами ответов в тесте. Верни только ответ, без пояснений, без кавычек, без координат, только сам ответ."
    )
    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
            ]}
        ],
        "max_tokens": 50
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        try:
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"Ошибка разбора ответа: {e}"
    else:
        return f"Ошибка OpenAI API: {response.status_code} {response.text}"

def find_text_coordinates_on_image(image_path, target_text, threshold=0.7):
    image = Image.open(image_path)
    data = pytesseract.image_to_data(image, lang='rus+eng', output_type=pytesseract.Output.DICT)
    lines = {}
    for i, text in enumerate(data['text']):
        if text.strip():
            y = data['top'][i]
            found = False
            for key in lines:
                if abs(key - y) < 10:
                    lines[key].append((data['left'][i], text.strip(), data['left'][i], data['top'][i], data['width'][i], data['height'][i]))
                    found = True
                    break
            if not found:
                lines[y] = [(data['left'][i], text.strip(), data['left'][i], data['top'][i], data['width'][i], data['height'][i])]
    matches = []
    print("\n[OCR DEBUG] Все строки (склеенные):")
    # Проверяем, английский ли ответ (только латиница и пробелы)
    is_english = re.fullmatch(r'[A-Za-z0-9 .,:;!?\-]+', target_text)
    for line in lines.values():
        line = sorted(line, key=lambda x: x[0])
        line_text = ' '.join([w[1] for w in line])
        clean_line = line_text
        # Если ответ на английском — игнорируем одиночный 'O' перед ответом
        if is_english:
            clean_line = re.sub(r'^O ', '', line_text)
        print(f"  '{line_text}'")
        ratio = SequenceMatcher(None, target_text.lower(), clean_line.lower()).ratio()
        if ratio > threshold or (len(target_text) <= 5 and target_text.lower() in clean_line.lower()):
            x = line[0][2] + (line[-1][2] + line[-1][4] - line[0][2]) // 2
            y = line[0][3] + line[0][5] // 2
            matches.append((x, y, ratio, line_text))
    if not matches:
        return None
    best_match = max(matches, key=lambda t: t[2])
    print(f"[OCR DEBUG] Лучшее совпадение: '{best_match[3]}' (коорд: {best_match[0]}, {best_match[1]}) с похожестью {best_match[2]:.2f}")
    return best_match[:2]

def find_nearest_circle_or_square(image_path, text_coords, search_radius=100):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        aspect = w / float(h)
        if 10 < w < 60 and 10 < h < 60 and 0.7 < aspect < 1.3:
            cx = x + w // 2
            cy = y + h // 2
            if (abs(cx - text_coords[0]) < search_radius) and (abs(cy - text_coords[1]) < search_radius):
                candidates.append((cx, cy, abs(cx - text_coords[0]) + abs(cy - text_coords[1])))
    if not candidates:
        return None
    best = min(candidates, key=lambda t: t[2])
    return best[:2]

def wait_and_click_on_coords(target_coords, tolerance=10, timeout=20):
    print(f"Ожидание наведения курсора в область {target_coords} (радиус {tolerance} пикселей)...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        x, y = pyautogui.position()
        if abs(x - target_coords[0]) <= tolerance and abs(y - target_coords[1]) <= tolerance:
            print(f"Курсор в нужной области! Кликаю по координатам: {x}, {y}")
            pyautogui.click(x, y)
            return True
        time.sleep(0.2)
    print("Время ожидания истекло, курсор не был наведен на нужную область.")
    return False

def get_area_by_right_ctrl():
    print('Выделите область: дважды нажмите Right Control — первый раз в начале области, второй раз в конце.')
    coords = []
    def on_press(key):
        if stop_flag.is_set():
            print('Остановка выбора области по запросу пользователя.')
            return False
        try:
            if key == keyboard.Key.ctrl_r:
                x, y = pyautogui.position()
                coords.append((x, y))
                print(f'Координаты {len(coords)}: {x}, {y}')
                if len(coords) == 2:
                    return False
        except Exception:
            pass
    with keyboard.Listener(on_press=on_press) as listener:
        while not stop_flag.is_set() and listener.running:
            listener.join(0.1)
        if stop_flag.is_set():
            print('Выход из выбора области по нажатию Home.')
            return None
    if len(coords) == 2:
        x1, y1 = coords[0]
        x2, y2 = coords[1]
        x, y = min(x1, x2), min(y1, y2)
        w, h = abs(x2 - x1), abs(y2 - y1)
        print(f'Область выбрана: x={x}, y={y}, w={w}, h={h}')
        return (x, y, w, h)
    else:
        print('Ошибка выделения области!')
        return None

def find_text_coordinates_on_image_soft(image_path, target_text, threshold=0.5):
    # Альтернативный способ: мягкий fuzzy + поиск по отдельным словам
    image = Image.open(image_path)
    data = pytesseract.image_to_data(image, lang='rus+eng', output_type=pytesseract.Output.DICT)
    lines = {}
    for i, text in enumerate(data['text']):
        if text.strip():
            y = data['top'][i]
            found = False
            for key in lines:
                if abs(key - y) < 10:
                    lines[key].append((data['left'][i], text.strip(), data['left'][i], data['top'][i], data['width'][i], data['height'][i]))
                    found = True
                    break
            if not found:
                lines[y] = [(data['left'][i], text.strip(), data['left'][i], data['top'][i], data['width'][i], data['height'][i])]
    matches = []
    print("\n[OCR DEBUG] (Альтернативный способ) Все строки (склеенные):")
    is_english = re.fullmatch(r'[A-Za-z0-9 .,:;!?\-]+', target_text)
    for line in lines.values():
        line = sorted(line, key=lambda x: x[0])
        line_text = ' '.join([w[1] for w in line])
        clean_line = line_text
        if is_english:
            clean_line = re.sub(r'^O ', '', line_text)
        print(f"  '{line_text}'")
        ratio = SequenceMatcher(None, target_text.lower(), clean_line.lower()).ratio()
        if ratio > threshold or (len(target_text) <= 5 and target_text.lower() in clean_line.lower()):
            x = line[0][2] + (line[-1][2] + line[-1][4] - line[0][2]) // 2
            y = line[0][3] + line[0][5] // 2
            matches.append((x, y, ratio, line_text))
    # Если не найдено по строкам — ищем по отдельным словам
    if not matches:
        print("[OCR DEBUG] (Альтернативный способ) Поиск по отдельным словам...")
        for i, text in enumerate(data['text']):
            if text.strip():
                ratio = SequenceMatcher(None, target_text.lower(), text.strip().lower()).ratio()
                if ratio > threshold or (len(target_text) <= 5 and target_text.lower() in text.strip().lower()):
                    x = data['left'][i] + data['width'][i] // 2
                    y = data['top'][i] + data['height'][i] // 2
                    matches.append((x, y, ratio, text.strip()))
    if not matches:
        return None
    best_match = max(matches, key=lambda t: t[2])
    print(f"[OCR DEBUG] (Альтернативный способ) Лучшее совпадение: '{best_match[3]}' (коорд: {best_match[0]}, {best_match[1]}) с похожестью {best_match[2]:.2f}")
    return best_match[:2]

def find_coords_by_letter(image_path, letter):
    image = Image.open(image_path)
    data = pytesseract.image_to_data(image, lang='rus+eng', output_type=pytesseract.Output.DICT)
    # Список возможных OCR-искажений для английских букв
    ocr_letter_map = {
        'a.': ['a.', 'A.', 'а.', 'А.', '0a', '0A', '0a.', '0A.'],
        'b.': ['b.', 'B.', 'ь.', 'Ь.', '0b', '0B', '0b.', '0B.'],
        'c.': ['c.', 'C.', 'с.', 'С.', '0c', '0C', '0c.', '0C.'],
        'd.': ['d.', 'D.', '9.', 'д.', 'Д.', '0d', '0D', '0d.', '0D.'],
        'e.': ['e.', 'E.', 'е.', 'Е.', '0e', '0E', '0e.', '0E.'],
        'f.': ['f.', 'F.', 'ф.', 'Ф.', '0f', '0F', '0f.', '0F.'],
        'g.': ['g.', 'G.', '9.', '0g', '0G', '0g.', '0G.'],
        # Можно добавить остальные буквы по аналогии
    }
    letter = letter.lower()
    candidates = ocr_letter_map.get(letter, [letter])
    print(f"[OCR DEBUG] Ищем варианты для '{letter}' среди: {candidates}")
    for i, text in enumerate(data['text']):
        t = text.strip().lower()
        for cand in candidates:
            if t.startswith(cand):
                x = data['left'][i] + data['width'][i] // 2
                y = data['top'][i] + data['height'][i] // 2
                print(f"[OCR DEBUG] Найден вариант по букве '{letter}': '{text.strip()}' (коорд: {x}, {y})")
                return (x, y)
    print(f"[OCR DEBUG] Не найден вариант по букве '{letter}' (искали среди: {candidates})")
    return None

def do_screenshot_and_ocr():
    print("Горячая клавиша (z) — выберите область для скриншота двумя нажатиями Right Control")
    region = get_area_by_right_ctrl()
    if not region:
        print('Область не выбрана!')
        return
    x1, y1, w, h = region
    screenshot = pyautogui.screenshot(region=(x1, y1, w, h))
    screenshot.save('test_screenshot.png')
    print(f'Скриншот области сохранён! Область: {region}')
    try:
        if not api_key:
            print('API-ключ не найден!')
            return
        gpt_response = send_image_to_gpt4o_answer_only(api_key, 'test_screenshot.png')
        print('Ответ от ChatGPT-4o:')
        print(gpt_response)
        answer = gpt_response.strip().split('\n')[0]
        print(f'== Поиск по букве ==')
        match = re.fullmatch(r'[a-zA-Zа-яА-Я]\.', answer.strip())
        if match:
            coords = find_coords_by_letter('test_screenshot.png', answer.strip())
            if coords:
                abs_coords = (x1 + coords[0], y1 + coords[1])
                print(f'Найдено по букве: относительные {coords}, абсолютные {abs_coords}')
                success = wait_and_click_on_coords(abs_coords)
                if success:
                    print('Успешно кликнули по буквенному варианту!')
                    return
                else:
                    print('Не удалось кликнуть по буквенному варианту, пробуем основной способ...')
        print(f'== Основной способ ==')
        coords = find_text_coordinates_on_image('test_screenshot.png', answer)
        if coords:
            abs_coords = (x1 + coords[0], y1 + coords[1])
            print(f'Найдено по основному способу: относительные {coords}, абсолютные {abs_coords}')
            success = wait_and_click_on_coords(abs_coords)
            if success:
                print('Успешно кликнули по основному способу!')
                return
            else:
                print('Не удалось кликнуть по основному способу, пробуем альтернативный...')
        print('== Альтернативный способ ==')
        coords = find_text_coordinates_on_image_soft('test_screenshot.png', answer)
        if coords:
            abs_coords = (x1 + coords[0], y1 + coords[1])
            print(f'Найдено по альтернативному способу: относительные {coords}, абсолютные {abs_coords}')
            success = wait_and_click_on_coords(abs_coords)
            if success:
                print('Успешно кликнули по альтернативному способу!')
                return
            else:
                print('Не удалось кликнуть по альтернативному способу, пробуем fallback...')
        print('== Fallback: OCR под курсором ==')
        wait_and_click_on_answer(answer)
        print('Попытка завершена.')
    except Exception as e:
        print(f'Ошибка при отправке в OpenAI: {e}')

def wait_and_click_on_answer(answer, timeout=15):
    print(f"Ожидание наведения на правильный вариант: '{answer}'")
    start_time = time.time()
    while time.time() - start_time < timeout:
        x, y = pyautogui.position()
        box_width = 400
        box_height = 50
        left = x - box_width // 2
        top = y - box_height // 2
        screenshot = pyautogui.screenshot(region=(left, top, box_width, box_height))
        try:
            text = pytesseract.image_to_string(screenshot, lang='rus+eng').strip()
            print(f"[OCR] Под курсором: '{text}'")
            if answer.lower() in text.lower():
                print(f"Обнаружен правильный вариант под курсором! Кликаю по координатам: {x}, {y}")
                pyautogui.click(x, y)
                return True
        except Exception as e:
            print(f"OCR ошибка: {e}")
        time.sleep(0.2)
    print("Время ожидания истекло, правильный вариант не найден под курсором.")
    return False

# Функция поиска текста только в заданной области изображения
# region = (x, y, w, h)
def find_text_coordinates_on_image_region(image_path, target_text, region, threshold=0.7):
    image = Image.open(image_path)
    cropped = image.crop((region[0], region[1], region[0]+region[2], region[1]+region[3]))
    data = pytesseract.image_to_data(cropped, lang='rus+eng', output_type=pytesseract.Output.DICT)
    lines = {}
    for i, text in enumerate(data['text']):
        if text.strip():
            y = data['top'][i]
            found = False
            for key in lines:
                if abs(key - y) < 10:
                    lines[key].append((data['left'][i], text.strip(), data['left'][i], data['top'][i], data['width'][i], data['height'][i]))
                    found = True
                    break
            if not found:
                lines[y] = [(data['left'][i], text.strip(), data['left'][i], data['top'][i], data['width'][i], data['height'][i])]
    matches = []
    is_english = re.fullmatch(r'[A-Za-z0-9 .,:;!?\-]+', target_text)
    for line in lines.values():
        line = sorted(line, key=lambda x: x[0])
        line_text = ' '.join([w[1] for w in line])
        clean_line = line_text
        if is_english:
            clean_line = re.sub(r'^O ', '', line_text)
        ratio = SequenceMatcher(None, target_text.lower(), clean_line.lower()).ratio()
        if ratio > threshold or (len(target_text) <= 5 and target_text.lower() in clean_line.lower()):
            # Координаты относительно обрезанного региона
            x = line[0][2] + (line[-1][2] + line[-1][4] - line[0][2]) // 2 + region[0]
            y = line[0][3] + line[0][5] // 2 + region[1]
            matches.append((x, y, ratio, line_text))
    if not matches:
        return None
    best_match = max(matches, key=lambda t: t[2])
    print(f"[OCR DEBUG] Лучшее совпадение (в области): '{best_match[3]}' (коорд: {best_match[0]}, {best_match[1]}) с похожестью {best_match[2]:.2f}")
    return best_match[:2]

def open_cmd_with_logs():
    import sys
    import os
    import subprocess

    exe_path = os.path.abspath(sys.argv[0])
    # Открываем новое окно cmd, которое запускает этот же exe с флагом --logs
    cmd = f'start "" cmd /k "{exe_path} --logs"'
    subprocess.Popen(cmd, shell=True)

def hotkey_listener():
    def on_activate_screenshot():
        do_screenshot_and_ocr()
    def on_activate_exit():
        print('Горячая клавиша (Home) — завершение работы.')
        stop_flag.set()
        QtWidgets.QApplication.quit()
    def on_activate_logs():
        print('Открываю окно с логами...')
        open_cmd_with_logs()
    hotkeys = {
        'z': on_activate_screenshot,
        '<home>': on_activate_exit,
        '<shift>+<space>': on_activate_logs,
    }
    with keyboard.GlobalHotKeys(hotkeys):
        while not stop_flag.is_set():
            threading.Event().wait(0.1)

def main():
    global api_key
    app = QtWidgets.QApplication(sys.argv)
    dialog = ApiKeyDialog()
    if dialog.exec_() == QtWidgets.QDialog.Accepted:
        api_key = dialog.api_key
        QtWidgets.QApplication.setQuitOnLastWindowClosed(False)
        hotkey_thread = threading.Thread(target=hotkey_listener, daemon=True)
        hotkey_thread.start()
        print('API-ключ получен, приложение работает в фоне...')
        app.exec_()
        stop_flag.set()
    else:
        sys.exit(0)

if __name__ == '__main__':
    # Если запущено с --logs, просто запускаем обычный режим с логами
    main()
