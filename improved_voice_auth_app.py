import sys
import os
import torch
import soundfile as sf
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                           QWidget, QLabel, QLineEdit, QStackedWidget, QHBoxLayout,
                           QFrame, QProgressBar, QMessageBox, QComboBox, QSlider, QListWidget,
                           QFileDialog, QTextEdit, QScrollArea)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor
from model import ECAPA_TDNN
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr
from speaker_verification import SpeakerVerifier
import librosa

class FileProcessingThread(QThread):
    """Отдельный поток для обработки аудио файла"""
    finished = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)
    
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        
    def run(self):
        try:
            # Чтение аудио файла
            audio_data, sample_rate = sf.read(self.file_path)
            
            # Преобразование в моно если стерео
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            # Ресемплинг если необходимо
            if sample_rate != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                
            self.finished.emit(audio_data)
        except Exception as e:
            self.error.emit(str(e))

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
        self.setWindowTitle("Voice Authentication System")
        self.setGeometry(100, 100, 800, 600)
        
        # Create main widget and layout
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        
        # Initialize UI components first
        self.init_ui()
        
        # Apply modern theme
        self.apply_modern_theme()
        
        # Initialize the speaker verification model
        self.verifier = SpeakerVerifier()
        
        # Initialize the ECAPA-TDNN model
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
        self.add_comparison_screen()
        
        # Начинаем с приветственного экрана
        self.stack.setCurrentIndex(0)
        
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
        
        # Основные кнопки
        register_btn = ModernButton("Регистрация нового пользователя")
        verify_btn = ModernButton("Верификация пользователя")
        
        register_btn.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        verify_btn.clicked.connect(lambda: self.stack.setCurrentIndex(2))
        
        layout.addWidget(register_btn)
        layout.addSpacing(20)
        layout.addWidget(verify_btn)
        
        # Разделитель
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("background-color: #BDBDBD;")
        layout.addSpacing(20)
        layout.addWidget(line)
        layout.addSpacing(20)
        
        # Кнопка сравнения (меньшего размера)
        compare_btn = QPushButton("Сравнение двух аудиофайлов")
        compare_btn.setMinimumHeight(35)  # Меньше чем у основных кнопок
        compare_btn.setStyleSheet("""
            QPushButton {
                background-color: #757575;
                border: none;
                border-radius: 5px;
                color: white;
                padding: 6px 12px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #616161;
            }
            QPushButton:pressed {
                background-color: #424242;
            }
        """)
        compare_btn.clicked.connect(lambda: self.stack.setCurrentIndex(5))  # Новый индекс для экрана сравнения
        layout.addWidget(compare_btn)
        
        self.stack.addWidget(welcome_widget)
        
    def add_registration_screen(self):
        """Создание экрана регистрации"""
        register_widget = QWidget()
        layout = QVBoxLayout(register_widget)
        
        # Заголовок
        title = QLabel("Регистрация нового пользователя")
        title.setStyleSheet("font-size: 20px; margin: 20px;")
        layout.addWidget(title)
        
        # Информация о множественном выборе файлов
        info_label = QLabel("Вы можете выбрать несколько аудиофайлов для создания более точного голосового профиля.\nВсе файлы будут объединены в один для обработки.")
        info_label.setStyleSheet("color: #757575; font-size: 13px; margin: 10px;")
        layout.addWidget(info_label)
        
        # Поля ввода
        self.username_input = ModernLineEdit()
        self.username_input.setPlaceholderText("Введите имя пользователя")
        layout.addWidget(self.username_input)
        
        # Список выбранных файлов
        self.selected_files_list = QListWidget()
        self.selected_files_list.setStyleSheet("""
            QListWidget {
                border: 2px solid #BDBDBD;
                border-radius: 5px;
                padding: 5px;
                background-color: white;
            }
            QListWidget::item {
                padding: 5px;
            }
        """)
        layout.addWidget(self.selected_files_list)
        
        # Кнопка выбора файлов
        select_files_btn = ModernButton("Выбрать аудио файлы")
        select_files_btn.clicked.connect(self.select_registration_files)
        layout.addWidget(select_files_btn)
        
        # Кнопка удаления выбранного файла
        remove_file_btn = ModernButton("Удалить выбранный файл")
        remove_file_btn.clicked.connect(self.remove_selected_file)
        layout.addWidget(remove_file_btn)
        
        # Кнопка регистрации
        register_btn = ModernButton("Зарегистрировать")
        register_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                border: none;
                border-radius: 5px;
                color: white;
                padding: 8px 16px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
        """)
        register_btn.clicked.connect(self.start_registration)
        layout.addWidget(register_btn)
        
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
        
        # Кнопка выбора файла
        select_file_btn = ModernButton("Выбрать аудио файл")
        select_file_btn.clicked.connect(self.select_verification_file)
        layout.addWidget(select_file_btn)
        
        # Путь к выбранному файлу
        self.verify_file_path_label = QLabel("Файл не выбран")
        self.verify_file_path_label.setStyleSheet("color: #757575;")
        layout.addWidget(self.verify_file_path_label)
        
        # Создаем скроллируемую область для результатов
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)
        
        # Контейнер для результатов
        self.results_container = QWidget()
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setAlignment(Qt.AlignTop)
        
        scroll_area.setWidget(self.results_container)
        layout.addWidget(scroll_area)
        
        # Кнопка назад
        back_btn = ModernButton("Назад")
        back_btn.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        layout.addWidget(back_btn)
        
        self.stack.addWidget(verify_widget)

    def add_settings_screen(self):
        """Экран настроек"""
        settings_widget = QWidget()
        layout = QVBoxLayout(settings_widget)
        
        # Настройки качества обработки
        quality_label = QLabel("Качество обработки")
        quality_label.setStyleSheet("font-size: 14px; margin: 10px;")
        layout.addWidget(quality_label)
        
        audio_quality = QComboBox()
        audio_quality.addItems(["Высокое качество", "Среднее качество", "Низкое качество"])
        audio_quality.setStyleSheet("""
            QComboBox {
                border: 2px solid #BDBDBD;
                border-radius: 5px;
                padding: 5px;
                min-height: 30px;
            }
            QComboBox:focus {
                border: 2px solid #2196F3;
            }
        """)
        layout.addWidget(audio_quality)
        
        # Настройки порога верификации
        threshold_label = QLabel("Порог верификации")
        threshold_label.setStyleSheet("font-size: 14px; margin: 10px;")
        layout.addWidget(threshold_label)
        
        threshold_slider = QSlider(Qt.Horizontal)
        threshold_slider.setRange(50, 90)
        threshold_slider.setValue(70)
        threshold_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #BDBDBD;
                height: 8px;
                background: #E0E0E0;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #2196F3;
                border: none;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)
        layout.addWidget(threshold_slider)
        
        # Значение порога
        threshold_value = QLabel("70%")
        threshold_value.setStyleSheet("color: #757575;")
        layout.addWidget(threshold_value)
        
        # Подключение сигналов
        threshold_slider.valueChanged.connect(
            lambda v: threshold_value.setText(f"{v}%")
        )
        
        # Кнопка сохранения настроек
        save_btn = ModernButton("Сохранить настройки")
        save_btn.clicked.connect(lambda: self.show_status("Настройки сохранены", "success"))
        layout.addWidget(save_btn)
        
        # Кнопка назад
        back_btn = ModernButton("Назад")
        back_btn.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        layout.addWidget(back_btn)
        
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
        
    def add_comparison_screen(self):
        """Создание экрана сравнения двух аудиофайлов"""
        compare_widget = QWidget()
        layout = QVBoxLayout(compare_widget)
        
        # Заголовок
        title = QLabel("Продвинутое сравнение голосов")
        title.setStyleSheet("font-size: 20px; margin: 20px;")
        layout.addWidget(title)
        
        # Первый файл
        first_file_layout = QHBoxLayout()
        self.first_file_label = QLabel("Первый файл не выбран")
        self.first_file_label.setStyleSheet("color: #757575;")
        select_first_btn = ModernButton("Выбрать первый файл")
        select_first_btn.clicked.connect(self.select_first_comparison_file)
        first_file_layout.addWidget(self.first_file_label)
        first_file_layout.addWidget(select_first_btn)
        layout.addLayout(first_file_layout)
        
        # Второй файл
        second_file_layout = QHBoxLayout()
        self.second_file_label = QLabel("Второй файл не выбран")
        self.second_file_label.setStyleSheet("color: #757575;")
        select_second_btn = ModernButton("Выбрать второй файл")
        select_second_btn.clicked.connect(self.select_second_comparison_file)
        second_file_layout.addWidget(self.second_file_label)
        second_file_layout.addWidget(select_second_btn)
        layout.addLayout(second_file_layout)
        
        # Кнопка сравнения
        compare_btn = ModernButton("Сравнить голоса")
        compare_btn.clicked.connect(self.compare_audio_files)
        layout.addWidget(compare_btn)
        
        # Область для графика
        self.figure = plt.figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Результат сравнения
        self.comparison_result = QTextEdit()
        self.comparison_result.setReadOnly(True)
        self.comparison_result.setStyleSheet("""
            QTextEdit {
                border: 2px solid #BDBDBD;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
                background-color: white;
            }
        """)
        layout.addWidget(self.comparison_result)
        
        # Кнопка назад
        back_btn = ModernButton("Назад")
        back_btn.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        layout.addWidget(back_btn)
        
        self.stack.addWidget(compare_widget)

    def select_first_comparison_file(self):
        """Выбор первого файла для сравнения"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите первый аудио файл",
            "",
            "Audio Files (*.wav *.mp3 *.ogg *.flac);;All Files (*.*)"
        )
        
        if file_path:
            self.first_file_label.setText(f"Выбран файл: {os.path.basename(file_path)}")
            self.first_comparison_file = file_path
            
    def select_second_comparison_file(self):
        """Выбор второго файла для сравнения"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите второй аудио файл",
            "",
            "Audio Files (*.wav *.mp3 *.ogg *.flac);;All Files (*.*)"
        )
        
        if file_path:
            self.second_file_label.setText(f"Выбран файл: {os.path.basename(file_path)}")
            self.second_comparison_file = file_path
            
    def compare_audio_files(self):
        """Сравнение двух аудиофайлов"""
        if not hasattr(self, 'first_comparison_file') or not hasattr(self, 'second_comparison_file'):
            self.comparison_result.setText("Выберите оба файла для сравнения")
            self.comparison_result.setStyleSheet("color: #F44336; font-size: 16px;")
            return
            
        try:
            # Обработка первого файла
            self.processing_thread = FileProcessingThread(self.first_comparison_file)
            self.processing_thread.finished.connect(self.handle_first_file_comparison)
            self.processing_thread.error.connect(lambda msg: self.comparison_result.setText(f"Ошибка: {msg}"))
            self.processing_thread.start()
            self.comparison_result.setText("Обработка файлов...")
            
        except Exception as e:
            self.comparison_result.setText(f"Ошибка при сравнении: {str(e)}")
            self.comparison_result.setStyleSheet("color: #F44336; font-size: 16px;")
            
    def handle_first_file_comparison(self, first_audio_data):
        """Обработка первого файла и запуск сравнения со вторым"""
        try:
            with torch.no_grad():
                first_audio_tensor = torch.FloatTensor(first_audio_data).cuda()
                first_audio_tensor = first_audio_tensor.unsqueeze(0)
                first_embedding = self.model(first_audio_tensor, aug=False)
                first_embedding = first_embedding.cpu().numpy()
                
            # Обработка второго файла
            self.processing_thread = FileProcessingThread(self.second_comparison_file)
            self.processing_thread.finished.connect(
                lambda second_audio_data: self.finish_comparison(first_audio_data, second_audio_data)
            )
            self.processing_thread.error.connect(lambda msg: self.comparison_result.setText(f"Ошибка: {msg}"))
            self.processing_thread.start()
            
        except Exception as e:
            self.comparison_result.setText(f"Ошибка при обработке первого файла: {str(e)}")
            self.comparison_result.setStyleSheet("color: #F44336; font-size: 16px;")
            
    def finish_comparison(self, first_audio_data, second_audio_data):
        """Завершение сравнения двух файлов"""
        try:
            # Save audio data to temporary files
            temp_first_file = "temp_first_audio.wav"
            temp_second_file = "temp_second_audio.wav"
            
            sf.write(temp_first_file, first_audio_data, 16000)
            sf.write(temp_second_file, second_audio_data, 16000)
            
            # Compare files using SpeakerVerifier
            score, prediction, detailed_metrics = self.verifier.verify_files(temp_first_file, temp_second_file)
            
            # Clean up temporary files
            os.remove(temp_first_file)
            os.remove(temp_second_file)
            
            # Calculate similarity percentage and convert to Python float
            similarity_percentage = float(score * 100)
            
            # Визуализация
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            # Построение графика сравнения
            metrics = ['Базовый\nсчет', 'Косинусное\nсходство', 'Корреляция\nПирсона', 
                      'Евклидово\nрасстояние', 'Манхэттенское\nрасстояние']
            values = [float(detailed_metrics['base_score']), 
                     float(detailed_metrics['cosine_similarity']),
                     float(detailed_metrics['pearson_correlation']),
                     float(np.exp(-detailed_metrics['euclidean_distance'])),
                     float(np.exp(-detailed_metrics['manhattan_distance'] / len(first_audio_data)))]
            
            bars = ax.bar(metrics, values)
            ax.set_ylim(0, 1)
            ax.set_title('Метрики сравнения голосов')
            
            # Добавление значений над столбцами
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom')
            
            self.canvas.draw()
            
            # Формирование подробного отчета
            result_text = "=== Результаты сравнения голосов ===\n\n"
            result_text += f"1. Базовый счет: {float(detailed_metrics['base_score']):.3f}\n"
            result_text += f"   • Основная оценка схожести голосов\n"
            result_text += f"   • Диапазон: от 0 до 1\n\n"
            
            result_text += f"2. Косинусное сходство: {float(detailed_metrics['cosine_similarity']):.3f}\n"
            result_text += f"   • Показывает схожесть направления векторов\n"
            result_text += f"   • Диапазон: от 0 до 1\n\n"
            
            result_text += f"3. Корреляция Пирсона: {float(detailed_metrics['pearson_correlation']):.3f}\n"
            result_text += f"   • Показывает линейную зависимость между векторами\n"
            result_text += f"   • Диапазон: от 0 до 1\n\n"
            
            result_text += f"4. Евклидово расстояние: {float(detailed_metrics['euclidean_distance']):.3f}\n"
            result_text += f"   • Показывает геометрическое расстояние между векторами\n"
            result_text += f"   • Меньше значение - больше схожесть\n\n"
            
            result_text += f"5. Манхэттенское расстояние: {float(detailed_metrics['manhattan_distance']):.3f}\n"
            result_text += f"   • Показывает сумму абсолютных разностей\n"
            result_text += f"   • Меньше значение - больше схожесть\n\n"
            
            result_text += "=== Заключение ===\n"
            if similarity_percentage > 85:  # Повышенный порог
                result_text += f"Голоса очень похожи! (Схожесть: {similarity_percentage:.1f}%)\n"
                result_text += "• Высокая вероятность, что это один и тот же голос\n"
                result_text += "• Рекомендуется дополнительная проверка при высоких требованиях к безопасности"
            elif similarity_percentage > 70:
                result_text += f"Голоса умеренно похожи (Схожесть: {similarity_percentage:.1f}%)\n"
                result_text += "• Есть признаки схожести, но требуется дополнительная проверка\n"
                result_text += "• Рекомендуется повторить запись"
            else:
                result_text += f"Голоса отличаются (Схожесть: {similarity_percentage:.1f}%)\n"
                result_text += "• Низкая вероятность, что это один и тот же голос\n"
                result_text += "• Рекомендуется повторить запись при необходимости"
            
            self.comparison_result.setText(result_text)
            
        except Exception as e:
            error_msg = f"Ошибка при сравнении: {str(e)}"
            print(f"\n{error_msg}")  # Вывод в консоль
            print(f"Тип ошибки: {type(e).__name__}")  # Вывод типа ошибки
            import traceback
            print("Полный стек ошибки:")
            print(traceback.format_exc())  # Вывод полного стека ошибки
            self.comparison_result.setText(error_msg)
            self.comparison_result.setStyleSheet("color: #F44336; font-size: 16px;")
            
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
        
    def select_registration_files(self):
        """Выбор файлов для регистрации"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Выберите аудио файлы",
            "",
            "Audio Files (*.wav *.mp3 *.ogg *.flac);;All Files (*.*)"
        )
        
        if file_paths:
            for file_path in file_paths:
                self.selected_files_list.addItem(file_path)
            self.show_status(f"Выбрано файлов: {len(file_paths)}", "info")
            
    def remove_selected_file(self):
        """Удаление выбранного файла из списка"""
        current_item = self.selected_files_list.currentItem()
        if current_item:
            self.selected_files_list.takeItem(self.selected_files_list.row(current_item))
            self.show_status(f"Осталось файлов: {self.selected_files_list.count()}", "info")
            
    def process_registration_file(self, file_path):
        """Обработка файла для регистрации"""
        if self.selected_files_list.count() == 0:
            self.show_status("Выберите хотя бы один аудио файл", "warning")
            return
            
        # Создаем временный файл для объединенного аудио
        temp_combined_file = "temp_combined_audio.wav"
        combined_audio = None
        
        try:
            # Объединяем все выбранные файлы
            for i in range(self.selected_files_list.count()):
                file_path = self.selected_files_list.item(i).text()
                audio_data, sample_rate = sf.read(file_path)
                
                # Преобразование в моно если стерео
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)
                
                if combined_audio is None:
                    combined_audio = audio_data
                else:
                    combined_audio = np.concatenate([combined_audio, audio_data])
            
            # Сохраняем объединенный файл
            sf.write(temp_combined_file, combined_audio, sample_rate)
            
            # Запускаем обработку объединенного файла
            self.processing_thread = FileProcessingThread(temp_combined_file)
            self.processing_thread.finished.connect(self.handle_registration_audio)
            self.processing_thread.error.connect(lambda msg: self.show_status(f"Ошибка: {msg}", "error"))
            self.processing_thread.start()
            self.show_status("Обработка объединенного файла...", "info")
            
        except Exception as e:
            self.show_status(f"Ошибка при объединении файлов: {str(e)}", "error")
            if os.path.exists(temp_combined_file):
                os.remove(temp_combined_file)
            
    def handle_registration_audio(self, audio_data):
        """Обработка аудио данных для регистрации"""
        try:
            username = self.username_input.text()
            
            # Сохранение аудио
            if not os.path.exists("voice_embeddings"):
                os.makedirs("voice_embeddings")
            
            audio_path = f"voice_embeddings/{username}.wav"
            sf.write(audio_path, audio_data, 16000)
            
            # Извлечение и сохранение эмбеддинга
            with torch.no_grad():
                audio_tensor = torch.FloatTensor(audio_data).cuda()
                audio_tensor = audio_tensor.unsqueeze(0)
                embedding = self.model(audio_tensor, aug=False)
                embedding = embedding.cpu().numpy()
                
            np.save(f"voice_embeddings/{username}.npy", embedding)
            
            self.show_status(f"Пользователь {username} успешно зарегистрирован", "success")
            
            # Очистка полей после успешной регистрации
            self.username_input.clear()
            self.selected_files_list.clear()
            
        except Exception as e:
            self.show_status(f"Ошибка при обработке файла: {str(e)}", "error")
            
    def select_verification_file(self):
        """Выбор файла для верификации"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите аудио файл",
            "",
            "Audio Files (*.wav *.mp3 *.ogg *.flac);;All Files (*.*)"
        )
        
        if file_path:
            self.verify_file_path_label.setText(f"Выбран файл: {os.path.basename(file_path)}")
            self.process_verification_file(file_path)
            
    def process_verification_file(self, file_path):
        """Обработка файла для верификации"""
        self.processing_thread = FileProcessingThread(file_path)
        self.processing_thread.finished.connect(self.verify_voice)
        self.processing_thread.error.connect(lambda msg: self.comparison_result.setText(f"Ошибка: {msg}"))
        self.processing_thread.start()
        self.comparison_result.setText("Обработка файла...")
        
    def verify_voice(self, audio_data):
        """Верификация записанного голоса"""
        try:
            # Save audio data to temporary file
            temp_file = "temp_verify_audio.wav"
            sf.write(temp_file, audio_data, 16000)
            
            # Compare with saved embeddings
            matches = []
            
            for file in os.listdir("voice_embeddings"):
                if file.endswith(".wav"):
                    saved_file = os.path.join("voice_embeddings", file)
                    score, prediction, detailed_metrics = self.verifier.verify_files(temp_file, saved_file)
                    
                    matches.append({
                        'username': file[:-4],  # Remove .wav extension
                        'score': float(score),
                        'metrics': detailed_metrics
                    })
            
            # Sort matches by score in descending order
            matches.sort(key=lambda x: x['score'], reverse=True)
            
            # Clean up temporary file
            os.remove(temp_file)
            
            # Clear previous results
            for i in reversed(range(self.results_layout.count())): 
                self.results_layout.itemAt(i).widget().setParent(None)
            
            # Add verification status
            status_label = QLabel()
            if matches:
                best_match = matches[0]
                if best_match['score'] > 0.7:  # Порог верификации
                    status_label.setText("✅ Верификация успешна!")
                    status_label.setStyleSheet("color: #4CAF50; font-size: 16px; font-weight: bold;")
                else:
                    status_label.setText("❌ Верификация не удалась")
                    status_label.setStyleSheet("color: #F44336; font-size: 16px; font-weight: bold;")
            else:
                status_label.setText("❌ Нет зарегистрированных пользователей для сравнения")
                status_label.setStyleSheet("color: #F44336; font-size: 16px; font-weight: bold;")
            
            self.results_layout.addWidget(status_label)
            
            # Add matches
            for i, match in enumerate(matches, 1):
                # Create container for each match
                match_container = QWidget()
                match_layout = QVBoxLayout(match_container)
                
                # Create header with username and score
                header = QWidget()
                header_layout = QHBoxLayout(header)
                
                username_label = QLabel(f"{i}. {match['username']}")
                username_label.setStyleSheet("font-size: 14px; font-weight: bold;")
                
                score_label = QLabel(f"Схожесть: {match['score']:.2%}")
                score_label.setStyleSheet("color: #2196F3; font-size: 14px;")
                
                details_btn = QPushButton("Подробнее")
                details_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #E0E0E0;
                        border: none;
                        border-radius: 3px;
                        padding: 5px 10px;
                        font-size: 12px;
                    }
                    QPushButton:hover {
                        background-color: #BDBDBD;
                    }
                """)
                
                header_layout.addWidget(username_label)
                header_layout.addWidget(score_label)
                header_layout.addWidget(details_btn)
                
                # Create details widget (initially hidden)
                details_widget = QWidget()
                details_layout = QVBoxLayout(details_widget)
                details_layout.setContentsMargins(20, 0, 0, 0)
                
                metrics_text = f"""
                • Базовый счет: {match['metrics']['base_score']:.3f}
                • Косинусное сходство: {match['metrics']['cosine_similarity']:.3f}
                • Корреляция Пирсона: {match['metrics']['pearson_correlation']:.3f}
                • Евклидово расстояние: {match['metrics']['euclidean_distance']:.3f}
                • Манхэттенское расстояние: {match['metrics']['manhattan_distance']:.3f}
                """
                details_label = QLabel(metrics_text)
                details_label.setStyleSheet("color: #757575; font-size: 13px;")
                details_layout.addWidget(details_label)
                
                details_widget.hide()
                
                # Connect button click to toggle details
                details_btn.clicked.connect(lambda checked, w=details_widget, b=details_btn: self.toggle_details(w, b))
                
                match_layout.addWidget(header)
                match_layout.addWidget(details_widget)
                
                self.results_layout.addWidget(match_container)
                
        except Exception as e:
            error_msg = f"Ошибка при верификации: {str(e)}"
            print(f"\n{error_msg}")  # Вывод в консоль
            print(f"Тип ошибки: {type(e).__name__}")  # Вывод типа ошибки
            import traceback
            print("Полный стек ошибки:")
            print(traceback.format_exc())  # Вывод полного стека ошибки
            
            # Clear previous results
            for i in reversed(range(self.results_layout.count())): 
                self.results_layout.itemAt(i).widget().setParent(None)
            
            error_label = QLabel(error_msg)
            error_label.setStyleSheet("color: #F44336; font-size: 14px;")
            self.results_layout.addWidget(error_label)

    def toggle_details(self, details_widget, button):
        """Переключение видимости деталей"""
        if details_widget.isVisible():
            details_widget.hide()
            button.setText("Подробнее")
        else:
            details_widget.show()
            button.setText("Скрыть")

    def start_registration(self):
        """Начало процесса регистрации"""
        if not self.username_input.text():
            self.show_status("Введите имя пользователя", "warning")
            return
            
        if self.selected_files_list.count() == 0:
            self.show_status("Выберите хотя бы один аудио файл", "warning")
            return
            
        self.process_registration_file(None)  # Передаем None, так как теперь используем список файлов

def main():
    app = QApplication(sys.argv)
    window = VoiceAuthApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
