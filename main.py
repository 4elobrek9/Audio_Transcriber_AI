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

# Перед запуском скрипта, в командной строке Windows выполните: chcp 65001
# Это установит кодировку UTF-8, чтобы символы отображались корректно.

CONFIG_FILE = "config.json"
SPEAKERS_FILE = "speakers.json"

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
VOSK_RATE = 16000
CHUNK = 4096


class SpeakerManager:
    """
    Класс для управления и идентификации спикеров на основе их голосовых векторов.
    Сохраняет векторы в файл, чтобы "запоминать" голоса между сеансами.
    """
    def __init__(self):
        self.speakers = {}
        self.next_speaker_id = 1
        self.tolerance = 0.5
        self.load_speakers()

    def load_speakers(self):
        """Загружает сохраненные векторы спикеров из файла."""
        if os.path.exists(SPEAKERS_FILE):
            try:
                with open(SPEAKERS_FILE, "r") as f:
                    self.speakers = json.load(f)
                    self.next_speaker_id = len(self.speakers) + 1
            except (IOError, json.JSONDecodeError):
                pass

    def save_speakers(self):
        """Сохраняет векторы спикеров в файл."""
        with open(SPEAKERS_FILE, "w") as f:
            json.dump(self.speakers, f)

    def get_speaker_label(self, spk_vector):
        """Идентифицирует спикера по голосовому вектору. Если спикер новый, добавляет его."""
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
    """Получает пути к моделям Vosk и спикеров из файла конфигурации или запрашивает у пользователя."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
                vosk_model_path = config.get("VOSK_MODEL_PATH")
                speaker_model_path = config.get("SPEAKER_MODEL_PATH")
                
                if vosk_model_path and os.path.isdir(vosk_model_path):
                    return vosk_model_path, speaker_model_path
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

                with open(CONFIG_FILE, "w") as f:
                    json.dump({
                        "VOSK_MODEL_PATH": vosk_model_path,
                        "SPEAKER_MODEL_PATH": speaker_model_path
                    }, f)
                return vosk_model_path, speaker_model_path
            except IOError as e:
                print(f"Ошибка при сохранении пути: {e}")
                return None, None
        else:
            print("Неверный путь. Пожалуйста, убедитесь, что папка существует.")


def find_vb_cable_device():
    """Находит виртуальное аудиоустройство VB-CABLE Output, которое используется как источник звука."""
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
        
        # Мы ищем VB-CABLE Output, так как он имеет входные каналы
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
    """
    Отправляет текст на обработку в LLM для структурирования и
    создания краткого изложения.
    """
    print("\nОбработка текста с помощью LLM...")
    
    # Prompt на русском, как вы просили
    system_prompt = (
        "Ты — профессиональный редактор и аналитик, твоя задача — взять сырую транскрипцию и превратить ее "
        "в связный, грамотный текст. Ты должен исправлять ошибки распознавания, сохраняя при этом основной "
        "смысл и контекст. Сделай текст более структурированным и лаконичным. Удали все лишние слова и "
        "паузы. Объедини короткие фразы в полные предложения и абзацы. В конце текста, после горизонтальной "
        "черты `---`, напиши краткую, но подробную выжимку 'Урок' из 4-15 предложений. В этой выжимке ты должен "
        "без воды и лишних слов объяснить, что было объяснено в тексте, какие ключевые идеи или шаги были представлены."
    )
    
    # Прямой вызов локального сервиса Ollama
    payload = {
        "model": "llama3:8b",
        "prompt": f"system: {system_prompt}\nuser: {text_to_process}",
        "stream": False
    }
    
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        response.raise_for_status() # Вызывает ошибку для плохих статусов
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при подключении к Ollama: {e}")
        return "Ошибка обработки текста. Пожалуйста, убедитесь, что Ollama запущен и модель llama3:8b загружена."

def get_folder_name_from_llm(text_to_process):
    """Генерирует название папки с помощью LLM."""
    print("Генерация названия папки с помощью AI...")
    
    system_prompt = (
        "Создай короткое и лаконичное название папки для транскрипции. Название должно быть "
        "максимум из 5 слов. Используй только русский язык. Исключи любые символы, кроме букв, цифр и пробелов."
    )
    
    payload = {
        "model": "llama3:8b",
        "prompt": f"system: {system_prompt}\nuser: {text_to_process}",
        "stream": False
    }
    
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        response.raise_for_status()
        llm_response = response.json()["response"].strip()
        # Санитизируем имя папки, заменяя недопустимые символы
        sanitized_name = re.sub(r'[\\/:*?"<>|]', '', llm_response)
        sanitized_name = sanitized_name.replace(" ", "_")
        return sanitized_name
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при подключении к Ollama для генерации имени папки: {e}")
        return datetime.datetime.now().strftime("ошибка_имени_%Y-%m-%d_%H-%M-%S")

def get_detailed_retelling_from_llm(text_to_process):
    """Создает подробный пересказ текста."""
    print("Создание подробного пересказа...")
    system_prompt = (
        "Ты — профессиональный пересказчик. Твоя задача — взять предоставленный текст, исправить все ошибки "
        "распознавания и оформить его в связный, подробный рассказ. Не удаляй важные детали, но сделай его "
        "более читабельным и логичным. Твоя цель — именно пересказать, а не просто суммировать."
    )
    payload = {
        "model": "llama3:8b",
        "prompt": f"system: {system_prompt}\nuser: {text_to_process}",
        "stream": False
    }
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при получении пересказа от Ollama: {e}")
        return "Ошибка при создании пересказа."

def get_questions_from_llm(text_to_process):
    """Генерирует 5-10 вопросов по тексту."""
    print("Генерация вопросов...")
    system_prompt = (
        "На основе предоставленного текста составь от 5 до 10 вопросов. Вопросы должны быть содержательными "
        "и побуждать к размышлениям о ключевых темах текста. Форматируй их в виде нумерованного списка."
    )
    payload = {
        "model": "llama3:8b",
        "prompt": f"system: {system_prompt}\nuser: {text_to_process}",
        "stream": False
    }
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при генерации вопросов от Ollama: {e}")
        return "Ошибка при создании вопросов."

def rewrite_last_transcription():
    """
    Находит последнюю созданную папку с транскрипцией,
    перезаписывает обработанный файл с помощью LLM.
    """
    print("\nЗапуск перекомпиляции последнего файла...")
    transcription_dir = "transcriptions"
    if not os.path.isdir(transcription_dir):
        print("Ошибка: Папка 'transcriptions' не найдена. Сначала нужно выполнить хотя бы одну запись.")
        return

    # Находим самую последнюю папку по времени создания
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
        
        # Запускаем LLM обработку
        processed_text = post_process_with_llm(full_transcription_text)
        retelling_text = get_detailed_retelling_from_llm(full_transcription_text)
        questions_text = get_questions_from_llm(full_transcription_text)

        final_output = f"## Обработанный текст и выжимка\n\n{processed_text}\n\n"
        final_output += f"## Подробный пересказ\n\n{retelling_text}\n\n"
        final_output += f"## Вопросы по теме\n\n{questions_text}"
        
        # Сохраняем результат
        with open(processed_file_path, "w", encoding="utf-8") as f:
            f.write(final_output)

        print(f"Файл '{processed_file_path}' успешно перезаписан.")
        
    except Exception as e:
        print(f"Критическая ошибка при перекомпиляции: {e}")


def transcribe_audio_stream(vb_cable_index, vosk_model_path, speaker_model_path):
    """Слушает аудиопоток с VB-CABLE и транскрибирует его."""
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
    
    try:
        # Открываем поток только для чтения с VB-CABLE Output
        input_stream = p.open(format=FORMAT,
                              channels=CHANNELS,
                              rate=RATE,
                              input=True,
                              input_device_index=vb_cable_index,
                              frames_per_buffer=CHUNK)

        print(f"Программа слушает VB-CABLE. Нажмите Ctrl+C для остановки.")
    except Exception as e:
        print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось открыть аудиопоток: {e}")
        return
    
    try:
        while True:
            data = input_stream.read(CHUNK, exception_on_overflow=False)
            
            # Resample для Vosk
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
                # Продолжаем выводить частичный результат
                partial_result = json.loads(recognizer.PartialResult())["partial"]
                if partial_result.strip() != "":
                    sys.stdout.write(partial_result + "\r")
                    sys.stdout.flush()
    
    except KeyboardInterrupt:
        print("\nПрограмма завершена.")
        
        # Обработка оставшихся данных перед выходом
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
        
        # --- Новая логика LLM-постобработки и генерации имени папки ---
        if full_transcription_text.strip():
            folder_name = get_folder_name_from_llm(full_transcription_text)
            transcription_folder = os.path.join("transcriptions", folder_name)
            os.makedirs(transcription_folder, exist_ok=True)
            
            raw_transcription_file_path = os.path.join(transcription_folder, "transcription_raw.txt")
            processed_transcription_file_path = os.path.join(transcription_folder, "transcription_processed.txt")
            
            # Сохраняем необработанный текст
            with open(raw_transcription_file_path, "w", encoding="utf-8") as f:
                f.write(full_transcription_text)

            # Обрабатываем и сохраняем текст, полученный от LLM
            processed_text = post_process_with_llm(full_transcription_text)
            
            # Получаем подробный пересказ
            retelling_text = get_detailed_retelling_from_llm(full_transcription_text)
            
            # Получаем вопросы
            questions_text = get_questions_from_llm(full_transcription_text)

            final_output = f"## Обработанный текст и выжимка\n\n{processed_text}\n\n"
            final_output += f"## Подробный пересказ\n\n{retelling_text}\n\n"
            final_output += f"## Вопросы по теме\n\n{questions_text}"
            
            with open(processed_transcription_file_path, "w", encoding="utf-8") as f:
                f.write(final_output)
            print(f"Транскрипции сохранены в папке '{transcription_folder}'.")

if __name__ == "__main__":
    vosk_model_path, speaker_model_path = get_model_paths()
    if vosk_model_path:
        vb_cable_index = find_vb_cable_device()
        if vb_cable_index is not None:
            transcribe_audio_stream(vb_cable_index, vosk_model_path, speaker_model_path)
    
    # Чтобы перекомпилировать последний файл, вызовите эту функцию
    # rewrite_last_transcription()
