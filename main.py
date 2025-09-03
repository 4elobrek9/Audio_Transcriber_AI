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
    def __init__(self):
        self.speakers = {}
        self.next_speaker_id = 1
        self.tolerance = 5.5
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
    
    # Создаем уникальную папку для этой сессии
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    transcription_folder = os.path.join("transcriptions", timestamp)
    os.makedirs(transcription_folder, exist_ok=True)
    raw_transcription_file_path = os.path.join(transcription_folder, "transcription_raw.txt")
    processed_transcription_file_path = os.path.join(transcription_folder, "transcription_processed.txt")

    try:
        # Открываем поток только для чтения с VB-CABLE Output
        input_stream = p.open(format=FORMAT,
                              channels=CHANNELS,
                              rate=RATE,
                              input=True,
                              input_device_index=vb_cable_index,
                              frames_per_buffer=CHUNK)

        print(f"Программа слушает VB-CABLE. Результат будет сохранен в папке '{transcription_folder}'. Нажмите Ctrl+C для остановки.")
    except Exception as e:
        print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось открыть аудиопоток: {e}")
        return
    
    full_transcription_text = ""
    last_transcribed_text = ""
    
    try:
        with open(raw_transcription_file_path, "a", encoding="utf-8") as f:
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
                        
                        f.write(text_to_write + "\n")
                        f.flush()
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
            with open(raw_transcription_file_path, "a", encoding="utf-8") as f:
                f.write(transcription + "\n")
            full_transcription_text += transcription + "\n"
            print(f"Результат (остаток): {transcription}")

    except Exception as e:
        print(f"\nОшибка: Программа завершилась с ошибкой: {e}")
    finally:
        if 'input_stream' in locals() and input_stream.is_active():
            input_stream.stop_stream()
            input_stream.close()
        p.terminate()
        
        # --- Новая логика LLM-постобработки ---
        if full_transcription_text.strip():
            processed_text = post_process_with_llm(full_transcription_text)
            with open(processed_transcription_file_path, "w", encoding="utf-8") as f:
                f.write(processed_text)
            print(f"Обработанный текст сохранен в: {processed_transcription_file_path}")

if __name__ == "__main__":
    vosk_model_path, speaker_model_path = get_model_paths()
    if vosk_model_path:
        vb_cable_index = find_vb_cable_device()
        if vb_cable_index is not None:
            transcribe_audio_stream(vb_cable_index, vosk_model_path, speaker_model_path)
