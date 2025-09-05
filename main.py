import pyaudio
from vosk import Model, KaldiRecognizer
import sys
import json
import os
import time
import datetime
from scipy.signal import resample
import numpy as np
import requests
import re
import cv2
import pyautogui
import pygetwindow as gw
from PIL import Image
import threading

CONFIG_FILE = "config.json"
SPEAKERS_FILE = "speakers.json"

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
VOSK_RATE = 16000
CHUNK = 4096

class SlideCapture:
    def __init__(self, window_title, save_path, interval=10, change_threshold=0.5):
        self.window_title = window_title
        self.save_path = save_path
        self.interval = interval
        self.change_threshold = change_threshold
        self.previous_screenshot = None
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def _capture_loop(self):
        while self.running:
            try:
                self._check_and_capture()
            except Exception as e:
                print(f"Ошибка при захвате скриншота: {e}")
            time.sleep(self.interval)

    def _get_browser_window(self):
        try:
            windows = gw.getWindowsWithTitle(self.window_title)
            if windows:
                return windows[0]
        except Exception as e:
            print(f"Ошибка при поиске окна браузера: {e}")
        return None

    def _take_screenshot(self, window):
        try:
            if window.isMinimized:
                window.restore()
            
            x, y, width, height = window.left, window.top, window.width, window.height
            screenshot = pyautogui.screenshot(region=(x, y, width, height))
            return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Ошибка при захвате скриншота: {e}")
            return None

    def _has_significant_change(self, img1, img2):
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        diff = cv2.absdiff(img1_gray, img2_gray)
        _, threshold_diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        changed_pixels = np.sum(threshold_diff) / 255
        total_pixels = img1_gray.size
        change_ratio = changed_pixels / total_pixels
        
        return change_ratio > self.change_threshold

    def _check_and_capture(self):
        window = self._get_browser_window()
        if not window:
            return

        screenshot = self._take_screenshot(window)
        if screenshot is None:
            return

        if self.previous_screenshot is not None:
            if self._has_significant_change(self.previous_screenshot, screenshot):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(self.save_path, f"slide_{timestamp}.png")
                cv2.imwrite(filename, screenshot)
                print(f"Сохранен скриншот слайда: {filename}")

        self.previous_screenshot = screenshot

class SpeakerManager:
    def __init__(self):
        self.speakers = {}
        self.next_speaker_id = 1
        self.tolerance = 0.5
        self.load_speakers()

    def load_speakers(self):
        if os.path.exists(SPEAKERS_FILE):
            try:
                with open(SPEAKERS_FILE, "r") as f:
                    self.speakers = json.load(f)
                    self.next_speaker_id = len(self.speakers) + 1
            except (IOError, json.JSONDecodeError):
                pass

    def save_speakers(self):
        with open(SPEAKERS_FILE, "w") as f:
            json.dump(self.speakers, f)

    def get_speaker_label(self, spk_vector):
        if not spk_vector:
            return "Неизвестный спикер"

        vector_array = np.array(spk_vector)

        for name, vector_data in self.speakers.items():
            stored_vector = np.array(vector_data["vector"])
            cosine_similarity = np.dot(vector_array, stored_vector) / (np.linalg.norm(vector_array) * np.linalg.norm(stored_vector))
            
            if cosine_similarity > 0.8:
                return name

        new_name = f"Спикер {self.next_speaker_id}"
        self.speakers[new_name] = {"vector": spk_vector}
        self.next_speaker_id += 1
        self.save_speakers()
        return new_name

def get_model_paths():
    browser_window_title = ""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
                vosk_model_path = config.get("VOSK_MODEL_PATH")
                speaker_model_path = config.get("SPEAKER_MODEL_PATH")
                browser_window_title = config.get("BROWSER_WINDOW_TITLE", "")
                
                if vosk_model_path and os.path.isdir(vosk_model_path):
                    return vosk_model_path, speaker_model_path, browser_window_title
                else:
                    print("Путь к модели Vosk в файле конфигурации недействителен.")
        except (IOError, json.JSONDecodeError):
            print("Ошибка чтения файла конфигурации.")
    
    while True:
        vosk_model_path = input("Введите путь к папке с моделью Vosk: ")
        if os.path.isdir(vosk_model_path):
            try:
                speaker_model_path = input("Введите путь к папке с моделью для спикеров (оставьте пустым, чтобы пропустить): ")
                if not os.path.isdir(speaker_model_path):
                    speaker_model_path = None
                    print("Путь к модели для спикеров недействителен, функция диаризации будет отключена.")

                browser_window_title = input("Введите заголовок окна браузера для захвата слайдов (оставьте пустым, чтобы пропустить): ")

                with open(CONFIG_FILE, "w") as f:
                    json.dump({
                        "VOSK_MODEL_PATH": vosk_model_path,
                        "SPEAKER_MODEL_PATH": speaker_model_path,
                        "BROWSER_WINDOW_TITLE": browser_window_title
                    }, f)
                return vosk_model_path, speaker_model_path, browser_window_title
            except IOError as e:
                print(f"Ошибка при сохранении пути: {e}")
                return None, None, ""
        else:
            print("Неверный путь. Пожалуйста, убедитесь, что папка существует.")

def find_vb_cable_device():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    
    vb_cable_index = -1
    print("---------------------------------")
    print("Доступные аудиоустройства:")
    print("---------------------------------")
    for i in range(0, numdevices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        device_name = device_info.get('name')
        input_channels = device_info.get('maxInputChannels')
        output_channels = device_info.get('maxOutputChannels')
        
        if "CABLE Output" in device_name and input_channels > 0:
            vb_cable_index = i
            print(f"Обнаружено устройство {i}: {device_name}")
            print(f"  Входных каналов: {input_channels}")
            break
        
    print("---------------------------------")
    p.terminate()

    if vb_cable_index == -1:
        print("Ошибка: Виртуальное устройство VB-CABLE (CABLE Output) не найдено.")
        print("Пожалуйста, убедитесь, что вы настроили Windows для перенаправления звука в 'CABLE Input' и перезапустите скрипт.")
        return None
    
    return vb_cable_index

def post_process_with_llm(text_to_process):
    print("\nОбработка текста с помощью LLM...")
    
    # 📝 СВЕРХПОДРОБНЫЙ ПРОМПТ ДЛЯ ПОСТ-ОБРАБОТКИ
    system_prompt = (
        "Ты — высококвалифицированный редактор и корректор, специализирующийся на учебных материалах и транскрипциях лекций. "
        "Твоя задача — взять сырую, необработанную транскрипцию устной речи на русском языке и превратить ее в безупречный, "
        "грамотный и профессиональный текст, который может быть использован как конспект или учебное пособие. "
        "Исходный текст содержит многочисленные ошибки, слова-паразиты, повторы, неполные предложения, "
        "особенности разговорной речи и возможные ошибки распознавания. "
        "Вся твоя работа должна быть выполнена исключительно на русском языке. Никаких вступлений или заключений, только обработанный текст. "
        "\n\n**Инструкции:**\n"
        "1.  **Полная очистка:** Внимательно прочитай весь текст. Удали все слова-паразиты ('ну', 'типа', 'короче'), междометия ('хм', 'ага'), "
        "повторы, которые не несут смысловой нагрузки ('очень-очень', 'я-я-я').\n"
        "2.  **Грамматика и пунктуация:** Исправь все грамматические, орфографические и пунктуационные ошибки. Расставь запятые, точки, тире и другие знаки препинания так, "
        "чтобы предложения стали правильными и легко читались.\n"
        "3.  **Структурирование:** Разбей текст на логические абзацы. Если в тексте есть перечисления или последовательности, "
        "отформатируй их в виде маркированных или нумерованных списков, чтобы улучшить читабельность.\n"
        "4.  **Связность:** Объедини неполные предложения в единый, связный текст. Переформулируй сложные или запутанные обороты, "
        "чтобы смысл стал предельно ясен. Текст должен течь плавно, как будто он изначально был написан, а не надиктован.\n"
        "5.  **Выжимка (резюме):** В самом конце текста, после горизонтальной черты `---`, создай новый раздел. "
        "В этом разделе предоставь краткую, но исчерпывающую выжимку всего материала. Этот раздел должен называться 'Ключевые выводы' "
        "и содержать от 4 до 15 предложений. Выжимка должна включать самые важные идеи, концепции, формулы и главные тезисы урока, "
        "позволяя быстро вспомнить его содержание. Не включай в выжимку второстепенные детали."
    )
    
    payload = {
        "model": "llama3:8b",
        "prompt": f"system: {system_prompt}\nuser: {text_to_process}",
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_ctx": 16384  # Увеличенное контекстное окно
        }
    }
    
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        response.raise_for_status()
        result = response.json()["response"]
        return result if result.strip() else "Не удалось обработать текст"
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при подключении к Ollama: {e}")
        return "Ошибка обработки текста. Пожалуйста, убедитесь, что Ollama запущен и модель llama3:8b загружена."

def get_folder_name_from_llm(text_to_process):
    print("Генерация названия папки с помощью AI...")
    
    # 📝 СВЕРХПОДРОБНЫЙ ПРОМПТ ДЛЯ НАЗВАНИЯ ПАПКИ
    system_prompt = (
        "Ты — специализированный генератор названий для файловой системы. "
        "Твоя единственная задача — придумать короткое, точное и удобное для использования название папки "
        "на основе предоставленной транскрипции урока. Название должно быть **коротким** (от 3 до 5 слов) и "
        "состоять **исключительно из кириллических букв, цифр и пробелов**. "
        "Крайне важно, чтобы в ответе было **только само название**, без каких-либо вводных фраз, кавычек, "
        "дополнительных пояснений, знаков препинания или спецсимволов. Название должно быть полностью на русском языке."
    )
    
    payload = {
        "model": "llama3:8b",
        "prompt": f"system: {system_prompt}\nuser: {text_to_process}",
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_ctx": 4096
        }
    }
    
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        response.raise_for_status()
        llm_response = response.json()["response"].strip()
        # Дополнительная проверка на всякий случай
        sanitized_name = re.sub(r'[^а-яА-Я0-9\s]', '', llm_response)
        sanitized_name = re.sub(r'\s+', '_', sanitized_name.strip())
        return sanitized_name if sanitized_name else "урок"
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при подключении к Ollama для генерации имени папки: {e}")
        return datetime.datetime.now().strftime("урок_%Y-%m-%d_%H-%M-%S")

def get_detailed_retelling_from_llm(text_to_process):
    print("Создание подробного пересказа...")
    # 📝 СВЕРХПОДРОБНЫЙ ПРОМПТ ДЛЯ ПЕРЕСКАЗА
    system_prompt = (
        "Ты — высококвалифицированный специалист по созданию подробных конспектов и пересказов. "
        "Твоя задача — создать максимально детализированный, полный и исчерпывающий пересказ "
        "предоставленного текста на русском языке. Твоя цель — сохранить каждую важную деталь, "
        "каждое объяснение, каждый пример и каждую концепцию, которые были в исходном материале. "
        "Пересказ должен быть понятным и логически структурированным, чтобы его мог понять "
        "даже человек, не знакомый с оригинальной транскрипцией. "
        "\n\n**Инструкции:**\n"
        "1.  **Полнота:** Ничего не упускай. Пересказ должен быть таким же длинным, как и оригинал, если не длиннее, "
        "за счет добавления связующих фраз для ясности. Включи все ключевые идеи, формулы, даты, имена и "
        "другие важные факты, которые присутствуют в тексте.\n"
        "2.  **Структура и ясность:** Организуй пересказ в логические абзацы. Используй заголовки, если это поможет "
        "разделить текст на смысловые блоки. Убедись, что переходы между абзацами и идеями плавные и естественные.\n"
        "3.  **Единственный ответ:** Отвечай только самим пересказом, без каких-либо вступлений или заключений. "
        "Весь ответ должен быть только на русском языке."
    )
    payload = {
        "model": "llama3:8b",
        "prompt": f"system: {system_prompt}\nuser: {text_to_process}",
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_ctx": 16384
        }
    }
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        response.raise_for_status()
        result = response.json()["response"]
        return result if result.strip() else "Не удалось создать пересказ"
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при получении пересказа от Ollama: {e}")
        return "Ошибка при создании пересказа."

def get_questions_from_llm(text_to_process):
    print("Генерация вопросов...")
    # 📝 СВЕРХПОДРОБНЫЙ ПРОМПТ ДЛЯ ВОПРОСОВ
    system_prompt = (
        "Ты — эксперт по составлению учебных вопросов. Твоя задача — создать 5-10 содержательных, "
        "развернутых вопросов на русском языке, которые требуют ответа, основанного на глубоком "
        "понимании материала, а не простого воспроизведения фактов. "
        "Вопросы должны охватывать ключевые концепции, причины, следствия, примеры и "
        "взаимосвязи, упомянутые в тексте. Избегай вопросов, на которые можно ответить "
        "односложно ('да' или 'нет'). "
        "Отвечай только списком вопросов, каждый с новой строки. Никаких вступлений, пояснений или "
        "заключений. Ответ должен быть полностью на русском языке."
    )
    payload = {
        "model": "llama3:8b",
        "prompt": f"system: {system_prompt}\nuser: {text_to_process}",
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_ctx": 8192
        }
    }
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        response.raise_for_status()
        result = response.json()["response"]
        return result if result.strip() else "Не удалось сгенерировать вопросы"
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при генерации вопросов от Ollama: {e}")
        return "Ошибка при создании вопросов."

def rewrite_last_transcription():
    print("\nЗапуск перекомпиляции последнего файла...")
    transcription_dir = "transcriptions"
    if not os.path.isdir(transcription_dir):
        print("Ошибка: Папка 'transcriptions' не найдена. Сначала нужно выполнить хотя бы одну запись.")
        return

    folders = [os.path.join(transcription_dir, d) for d in os.listdir(transcription_dir) if os.path.isdir(os.path.join(transcription_dir, d))]
    if not folders:
        print("Ошибка: Папка 'transcriptions' пуста. Сначала нужно выполнить хотя бы одну запись.")
        return
    
    latest_folder = max(folders, key=os.path.getctime)
    raw_file_path = os.path.join(latest_folder, "transcription_raw.txt")
    processed_file_path = os.path.join(latest_folder, "transcription_processed.txt")
    
    if not os.path.exists(raw_file_path):
        print(f"Ошибка: Исходный файл '{raw_file_path}' не найден. Перекомпиляция невозможна.")
        return
        
    try:
        with open(raw_file_path, "r", encoding="utf-8") as f:
            full_transcription_text = f.read()

        print(f"Найден исходный файл: '{raw_file_path}'")
        
        processed_text = post_process_with_llm(full_transcription_text)
        retelling_text = get_detailed_retelling_from_llm(full_transcription_text)
        questions_text = get_questions_from_llm(full_transcription_text)

        final_output = f"## Обработанный текст и выжимка\n\n{processed_text}\n\n"
        final_output += f"## Подробный пересказ\n\n{retelling_text}\n\n"
        final_output += f"## Вопросы по теме\n\n{questions_text}"
        
        with open(processed_file_path, "w", encoding="utf-8") as f:
            f.write(final_output)

        print(f"Файл '{processed_file_path}' успешно перезаписан.")
        
    except Exception as e:
        print(f"Критическая ошибка при перекомпиляции: {e}")

def transcribe_audio_stream(vb_cable_index, vosk_model_path, speaker_model_path, browser_window_title):
    try:
        model = Model(vosk_model_path)
        speaker_model = Model(speaker_model_path) if speaker_model_path else None
    except Exception as e:
        print(f"ОШИБКА: Не удалось загрузить модель Vosk. Убедитесь, что пути корректны: {e}")
        return

    recognizer = KaldiRecognizer(model, VOSK_RATE)
    if speaker_model:
        recognizer.SetSpkModel(speaker_model)
    
    p = pyaudio.PyAudio()
    speaker_manager = SpeakerManager()
    
    full_transcription_text = ""
    last_transcribed_text = ""
    
    slide_capture = None
    if browser_window_title:
        slide_capture = SlideCapture(browser_window_title, "", interval=10)
    
    try:
        input_stream = p.open(format=FORMAT,
                              channels=CHANNELS,
                              rate=RATE,
                              input=True,
                              input_device_index=vb_cable_index,
                              frames_per_buffer=CHUNK)

        print(f"Программа слушает VB-CABLE. Нажмите Ctrl+C для остановки.")
        
        if slide_capture:
            slide_capture.start()
            
    except Exception as e:
        print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось открыть аудиопоток: {e}")
        return
    
    try:
        while True:
            data = input_stream.read(CHUNK, exception_on_overflow=False)
            
            resampled_data = resample(np.frombuffer(data, dtype=np.int16), int(CHUNK * VOSK_RATE / RATE)).astype(np.int16).tobytes()
            
            if recognizer.AcceptWaveform(resampled_data):
                result_data = json.loads(recognizer.Result())
                transcription = result_data.get("text", "")
                speaker_vector = result_data.get("spk")
                
                if transcription and transcription.strip() != "" and transcription.strip() != last_transcribed_text:
                    text_to_write = transcription
                    if speaker_vector:
                        speaker_label = speaker_manager.get_speaker_label(speaker_vector)
                        text_to_write = f"[{speaker_label}]: {transcription}"
                    
                    full_transcription_text += text_to_write + "\n"
                    last_transcribed_text = transcription.strip()
                    print(f"Результат: {text_to_write}")
            else:
                partial_result = json.loads(recognizer.PartialResult())["partial"]
                if partial_result.strip() != "":
                    sys.stdout.write(partial_result + "\r")
                    sys.stdout.flush()
    
    except KeyboardInterrupt:
        print("\nПрограмма завершена.")
        
        final_result = json.loads(recognizer.FinalResult())
        transcription = final_result.get("text", "")
        if transcription and transcription.strip() != "":
            full_transcription_text += transcription + "\n"
            print(f"Результат (остаток): {transcription}")

    except Exception as e:
        print(f"\nОшибка: Программа завершилась с ошибкой: {e}")
    finally:
        if 'input_stream' in locals() and input_stream.is_active():
            input_stream.stop_stream()
            input_stream.close()
        p.terminate()
        
        if slide_capture:
            slide_capture.stop()
        
        if full_transcription_text.strip(): 
            folder_name = get_folder_name_from_llm(full_transcription_text)
            transcription_folder = os.path.join("transcriptions", folder_name)
            os.makedirs(transcription_folder, exist_ok=True)
            
            if slide_capture:
                slide_capture.save_path = transcription_folder
                slide_capture._check_and_capture()
            
            raw_transcription_file_path = os.path.join(transcription_folder, "transcription_raw.txt")
            processed_transcription_file_path = os.path.join(transcription_folder, "transcription_processed.txt")
            
            with open(raw_transcription_file_path, "w", encoding="utf-8") as f:
                f.write(full_transcription_text)

            processed_text = post_process_with_llm(full_transcription_text)
            retelling_text = get_detailed_retelling_from_llm(full_transcription_text)
            questions_text = get_questions_from_llm(full_transcription_text)

            final_output = f"## Обработанный текст и выжимка\n\n{processed_text}\n\n"
            final_output += f"## Подробный пересказ\n\n{retelling_text}\n\n"
            final_output += f"## Вопросы по теме\n\n{questions_text}"
            
            with open(processed_transcription_file_path, "w", encoding="utf-8") as f:
                f.write(final_output)
            print(f"Транскрипции сохранены в папке '{transcription_folder}'.")

if __name__ == "__main__":
    vosk_model_path, speaker_model_path, browser_window_title = get_model_paths()
    if vosk_model_path:
        vb_cable_index = find_vb_cable_device()
        if vb_cable_index is not None:
            transcribe_audio_stream(vb_cable_index, vosk_model_path, speaker_model_path, browser_window_title)