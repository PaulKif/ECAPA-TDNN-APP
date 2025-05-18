import sys
import os
import torch
import sounddevice as sd
import soundfile as sf
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                           QWidget, QLabel, QLineEdit, QStackedWidget, QHBoxLayout,
                           QFrame, QProgressBar, QMessageBox, QComboBox, QSlider, QListWidget)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor
from model import ECAPA_TDNN
from cryptography.fernet import Fernet

class RecordingThread(QThread):
    """Отдельный поток для записи звука"""
    finished = pyqtSignal(np.ndarray)
    
    def __init__(self, duration, sample_rate):
        super().__init__()
        self.duration = duration
        self.sample_rate = sample_rate
        
    def run(self):
        recording = sd.rec(int(self.duration * self.sample_rate),
                         samplerate=self.sample_rate,
                         channels=1,
                         dtype='float32')
        sd.wait()
        self.finished.emit(recording)

class ModernButton(QPushButton):
    """Кастомная кнопка с современным дизайном"""
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setMinimumHeight(40)
        self.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                border: none;
                border-radius: 5px;
                color: white;
                padding: 8px 16px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
        """)

class ModernLineEdit(QLineEdit):
    """Кастомное поле ввода с современным дизайном"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(40)
        self.setStyleSheet("""
            QLineEdit {
                border: 2px solid #BDBDBD;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 2px solid #2196F3;
            }
        """)

class VoiceAuthApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.init_model()
        
    def init_model(self):
        """Инициализация модели ECAPA-TDNN"""
        try:
            self.model = ECAPA_TDNN(C=1024).cuda()
            self.model.eval()
            self.show_status("Модель успешно загружена", "success")
        except Exception as e:
            self.show_status(f"Ошибка загрузки модели: {str(e)}", "error")
            
    def init_ui(self):
        """Инициализация пользовательского интерфейса"""
        self.setWindowTitle("Система голосовой аутентификации")
        self.setMinimumSize(800, 600)
        
        # Создаем центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Основной layout
        main_layout = QVBoxLayout(central_widget)
        
        # Создаем стек виджетов для разных экранов
        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack)
        
        # Добавляем экраны
        self.add_welcome_screen()
        self.add_registration_screen()
        self.add_verification_screen()
        self.add_settings_screen()
        self.add_user_management_screen()
        
        # Начинаем с приветственного экрана
        self.stack.setCurrentIndex(0)
        
        # Применяем современную тему
        self.apply_modern_theme()
        
    def add_welcome_screen(self):
        """Создание приветственного экрана"""
        welcome_widget = QWidget()
        layout = QVBoxLayout(welcome_widget)
        layout.setAlignment(Qt.AlignCenter)
        
        # Заголовок
        title = QLabel("Добро пожаловать в систему\nголосовой аутентификации")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; margin: 20px;")
        layout.addWidget(title)
        
        # Кнопки
        register_btn = ModernButton("Регистрация нового пользователя")
        verify_btn = ModernButton("Верификация пользователя")
        
        register_btn.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        verify_btn.clicked.connect(lambda: self.stack.setCurrentIndex(2))
        
        layout.addWidget(register_btn)
        layout.addSpacing(20)
        layout.addWidget(verify_btn)
        
        self.stack.addWidget(welcome_widget)
        
    def add_registration_screen(self):
        """Создание экрана регистрации"""
        register_widget = QWidget()
        layout = QVBoxLayout(register_widget)
        
        # Заголовок
        title = QLabel("Регистрация нового пользователя")
        title.setStyleSheet("font-size: 20px; margin: 20px;")
        layout.addWidget(title)
        
        # Поля ввода
        self.username_input = ModernLineEdit()
        self.username_input.setPlaceholderText("Введите имя пользователя")
        layout.addWidget(self.username_input)
        
        # Кнопки записи
        record_btn = ModernButton("Начать запись голоса")
        record_btn.clicked.connect(self.start_recording)
        layout.addWidget(record_btn)
        
        # Прогресс бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #BDBDBD;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Статус
        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: #757575;")
        layout.addWidget(self.status_label)
        
        # Кнопка назад
        back_btn = ModernButton("Назад")
        back_btn.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        layout.addWidget(back_btn)
        
        self.stack.addWidget(register_widget)
        
    def add_verification_screen(self):
        """Создание экрана верификации"""
        verify_widget = QWidget()
        layout = QVBoxLayout(verify_widget)
        
        # Заголовок
        title = QLabel("Верификация пользователя")
        title.setStyleSheet("font-size: 20px; margin: 20px;")
        layout.addWidget(title)
        
        # Кнопка записи
        verify_btn = ModernButton("Начать верификацию")
        verify_btn.clicked.connect(self.start_verification)
        layout.addWidget(verify_btn)
        
        # Прогресс верификации
        self.verify_progress = QProgressBar()
        self.verify_progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #BDBDBD;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
            }
        """)
        layout.addWidget(self.verify_progress)
        
        # Результат верификации
        self.verify_result = QLabel()
        self.verify_result.setStyleSheet("font-size: 16px; margin: 20px;")
        layout.addWidget(self.verify_result)
        
        # Кнопка назад
        back_btn = ModernButton("Назад")
        back_btn.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        layout.addWidget(back_btn)
        
        self.stack.addWidget(verify_widget)
        
    def add_settings_screen(self):
        """Экран настроек"""
        settings_widget = QWidget()
        layout = QVBoxLayout(settings_widget)
        
        # Настройки качества записи
        audio_quality = QComboBox()
        audio_quality.addItems(["Высокое качество", "Среднее качество", "Низкое качество"])
        
        # Настройки порога верификации
        threshold_slider = QSlider(Qt.Horizontal)
        threshold_slider.setRange(50, 90)
        threshold_slider.setValue(70)
        
        # Выбор устройства записи
        devices = sd.query_devices()
        device_selector = QComboBox()
        device_selector.addItems([d['name'] for d in devices])
        
        layout.addWidget(QLabel("Качество записи"))
        layout.addWidget(audio_quality)
        layout.addWidget(QLabel("Порог верификации"))
        layout.addWidget(threshold_slider)
        layout.addWidget(QLabel("Устройство записи"))
        layout.addWidget(device_selector)
        
        self.stack.addWidget(settings_widget)

    def add_user_management_screen(self):
        """Экран управления пользователями"""
        management_widget = QWidget()
        layout = QVBoxLayout(management_widget)
        
        # Список зарегистрированных пользователей
        user_list = QListWidget()
        for file in os.listdir("voice_embeddings"):
            if file.endswith(".npy"):
                user_list.addItem(file[:-4])
        
        # Кнопки управления
        delete_btn = ModernButton("Удалить пользователя")
        rename_btn = ModernButton("Переименовать")
        update_btn = ModernButton("Обновить голосовой профиль")
        
        layout.addWidget(user_list)
        layout.addWidget(delete_btn)
        layout.addWidget(rename_btn)
        layout.addWidget(update_btn)
        
        self.stack.addWidget(management_widget)
        
    def apply_modern_theme(self):
        """Применение современной темы оформления"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #FAFAFA;
            }
            QLabel {
                color: #212121;
            }
            QWidget {
                font-family: 'Segoe UI', Arial;
            }
        """)
        
    def show_status(self, message, status_type="info"):
        """Отображение статусных сообщений"""
        colors = {
            "success": "#4CAF50",
            "error": "#F44336",
            "info": "#2196F3",
            "warning": "#FFC107"
        }
        self.status_label.setStyleSheet(f"color: {colors.get(status_type, '#757575')};")
        self.status_label.setText(message)
        
    def start_recording(self):
        """Начало записи голоса"""
        if not self.username_input.text():
            self.show_status("Введите имя пользователя", "warning")
            return
            
        self.recording_thread = RecordingThread(duration=3, sample_rate=16000)
        self.recording_thread.finished.connect(self.process_recording)
        
        # Настройка прогресс бара
        self.progress_bar.setRange(0, 100)
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_value = 0
        
        # Старт записи
        self.recording_thread.start()
        self.progress_timer.start(30)  # Обновление каждые 30мс
        self.show_status("Идет запись...", "info")
        
    def update_progress(self):
        """Обновление прогресс бара"""
        self.progress_value += 1
        self.progress_bar.setValue(self.progress_value)
        
        if self.progress_value >= 100:
            self.progress_timer.stop()
            
    def process_recording(self, recording):
        """Обработка записанного голоса"""
        try:
            username = self.username_input.text()
            
            # Сохранение аудио
            if not os.path.exists("voice_embeddings"):
                os.makedirs("voice_embeddings")
            
            audio_path = f"voice_embeddings/{username}.wav"
            sf.write(audio_path, recording, 16000)
            
            # Извлечение и сохранение эмбеддинга
            with torch.no_grad():
                audio_tensor = torch.FloatTensor(recording).cuda()
                audio_tensor = audio_tensor.unsqueeze(0)
                embedding = self.model(audio_tensor, aug=False)
                embedding = embedding.cpu().numpy()
                
            np.save(f"voice_embeddings/{username}.npy", embedding)
            
            self.show_status(f"Пользователь {username} успешно зарегистрирован", "success")
            
        except Exception as e:
            self.show_status(f"Ошибка при обработке записи: {str(e)}", "error")
            
    def start_verification(self):
        """Начало процесса верификации"""
        self.recording_thread = RecordingThread(duration=3, sample_rate=16000)
        self.recording_thread.finished.connect(self.verify_voice)
        
        # Настройка прогресс бара
        self.verify_progress.setRange(0, 100)
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_verify_progress)
        self.progress_value = 0
        
        # Старт записи
        self.recording_thread.start()
        self.progress_timer.start(30)
        self.verify_result.setText("Идет запись голоса...")
        
    def update_verify_progress(self):
        """Обновление прогресс бара верификации"""
        self.progress_value += 1
        self.verify_progress.setValue(self.progress_value)
        
        if self.progress_value >= 100:
            self.progress_timer.stop()
            
    def verify_voice(self, recording):
        """Верификация записанного голоса"""
        try:
            # Извлечение эмбеддинга для проверяемого голоса
            with torch.no_grad():
                audio_tensor = torch.FloatTensor(recording).cuda()
                audio_tensor = audio_tensor.unsqueeze(0)
                test_embedding = self.model(audio_tensor, aug=False)
                test_embedding = test_embedding.cpu().numpy()
            
            # Сравнение с сохраненными эмбеддингами
            best_match = None
            best_score = -1
            
            for file in os.listdir("voice_embeddings"):
                if file.endswith(".npy"):
                    saved_embedding = np.load(f"voice_embeddings/{file}")
                    score = np.dot(test_embedding, saved_embedding.T)
                    if score > best_score:
                        best_score = score
                        best_match = file[:-4]
            
            if best_score > 0.7:
                self.verify_result.setText(f"Верификация успешна!\nПользователь: {best_match}")
                self.verify_result.setStyleSheet("color: #4CAF50; font-size: 16px;")
            else:
                self.verify_result.setText("Верификация не удалась.\nПользователь не распознан.")
                self.verify_result.setStyleSheet("color: #F44336; font-size: 16px;")
                
        except Exception as e:
            self.verify_result.setText(f"Ошибка при верификации: {str(e)}")
            self.verify_result.setStyleSheet("color: #F44336; font-size: 16px;")

    def encrypt_embedding(self, embedding, key):
        """Шифрование эмбеддинга"""
        f = Fernet(key)
        return f.encrypt(embedding.tobytes())

    def decrypt_embedding(self, encrypted_embedding, key):
        """Дешифрование эмбеддинга"""
        f = Fernet(key)
        return np.frombuffer(f.decrypt(encrypted_embedding))

def main():
    app = QApplication(sys.argv)
    window = VoiceAuthApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()