import sys
import os
import torch
import soundfile as sf
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                           QWidget, QLabel, QLineEdit, QStackedWidget, QHBoxLayout,
                           QFrame, QProgressBar, QMessageBox, QComboBox, QSlider, QListWidget,
                           QFileDialog, QTextEdit)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor
from model import ECAPA_TDNN
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr

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
                # Здесь можно добавить ресемплинг если нужно
                pass
                
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
        self.add_comparison_screen()
        
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
        
        # Поля ввода
        self.username_input = ModernLineEdit()
        self.username_input.setPlaceholderText("Введите имя пользователя")
        layout.addWidget(self.username_input)
        
        # Кнопка выбора файла
        select_file_btn = ModernButton("Выбрать аудио файл")
        select_file_btn.clicked.connect(self.select_registration_file)
        layout.addWidget(select_file_btn)
        
        # Путь к выбранному файлу
        self.file_path_label = QLabel("Файл не выбран")
        self.file_path_label.setStyleSheet("color: #757575;")
        layout.addWidget(self.file_path_label)
        
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
                lambda second_audio_data: self.finish_comparison(first_embedding, second_audio_data)
            )
            self.processing_thread.error.connect(lambda msg: self.comparison_result.setText(f"Ошибка: {msg}"))
            self.processing_thread.start()
            
        except Exception as e:
            self.comparison_result.setText(f"Ошибка при обработке первого файла: {str(e)}")
            self.comparison_result.setStyleSheet("color: #F44336; font-size: 16px;")
            
    def finish_comparison(self, first_embedding, second_audio_data):
        """Завершение сравнения двух файлов"""
        try:
            with torch.no_grad():
                second_audio_tensor = torch.FloatTensor(second_audio_data).cuda()
                second_audio_tensor = second_audio_tensor.unsqueeze(0)
                second_embedding = self.model(second_audio_tensor, aug=False)
                second_embedding = second_embedding.cpu().numpy()
            
            # Нормализация эмбеддингов
            first_embedding = first_embedding / np.linalg.norm(first_embedding)
            second_embedding = second_embedding / np.linalg.norm(second_embedding)
            
            # Вычисление различных метрик
            # 1. Косинусное сходство
            cosine_sim = 1 - cosine(first_embedding.flatten(), second_embedding.flatten())
            
            # 2. Евклидово расстояние (нормализованное)
            euclidean_dist = euclidean(first_embedding.flatten(), second_embedding.flatten())
            euclidean_sim = np.exp(-euclidean_dist)  # Экспоненциальное преобразование
            
            # 3. Корреляция Пирсона
            pearson_corr, _ = pearsonr(first_embedding.flatten(), second_embedding.flatten())
            pearson_sim = (pearson_corr + 1) / 2  # Нормализация к [0,1]
            
            # 4. Манхэттенское расстояние (нормализованное)
            manhattan_dist = np.sum(np.abs(first_embedding - second_embedding))
            manhattan_sim = np.exp(-manhattan_dist / len(first_embedding.flatten()))
            
            # 5. Косинусное сходство по частям (более строгая метрика)
            chunk_size = len(first_embedding.flatten()) // 4
            chunk_sims = []
            for i in range(0, len(first_embedding.flatten()), chunk_size):
                chunk1 = first_embedding.flatten()[i:i+chunk_size]
                chunk2 = second_embedding.flatten()[i:i+chunk_size]
                if len(chunk1) == len(chunk2):
                    chunk_sim = 1 - cosine(chunk1, chunk2)
                    chunk_sims.append(chunk_sim)
            chunk_sim = np.mean(chunk_sims)
            
            # Комбинированная оценка с весами
            weights = {
                'cosine': 0.3,
                'euclidean': 0.2,
                'pearson': 0.2,
                'manhattan': 0.15,
                'chunk': 0.15
            }
            
            combined_score = (
                weights['cosine'] * cosine_sim +
                weights['euclidean'] * euclidean_sim +
                weights['pearson'] * pearson_sim +
                weights['manhattan'] * manhattan_sim +
                weights['chunk'] * chunk_sim
            )
            
            # Преобразование в проценты с нелинейным масштабированием
            similarity_percentage = (np.tanh(combined_score * 2 - 1) + 1) * 50
            
            # Визуализация
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            # Построение графика сравнения
            metrics = ['Косинусное\nсходство', 'Евклидово\nсходство', 'Корреляция\nПирсона', 
                      'Манхэттенское\nсходство', 'Сходство по\nчастям', 'Комбинированная\nоценка']
            values = [cosine_sim, euclidean_sim, pearson_sim, manhattan_sim, chunk_sim, combined_score]
            
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
            result_text += f"1. Косинусное сходство: {cosine_sim:.3f}\n"
            result_text += f"   • Показывает схожесть направления векторов\n"
            result_text += f"   • Диапазон: от 0 до 1\n\n"
            
            result_text += f"2. Евклидово сходство: {euclidean_sim:.3f}\n"
            result_text += f"   • Показывает близость векторов в пространстве\n"
            result_text += f"   • Диапазон: от 0 до 1\n\n"
            
            result_text += f"3. Корреляция Пирсона: {pearson_sim:.3f}\n"
            result_text += f"   • Показывает линейную зависимость между векторами\n"
            result_text += f"   • Диапазон: от 0 до 1\n\n"
            
            result_text += f"4. Манхэттенское сходство: {manhattan_sim:.3f}\n"
            result_text += f"   • Показывает схожесть по абсолютным значениям\n"
            result_text += f"   • Диапазон: от 0 до 1\n\n"
            
            result_text += f"5. Сходство по частям: {chunk_sim:.3f}\n"
            result_text += f"   • Показывает локальную схожесть в разных частях голоса\n"
            result_text += f"   • Диапазон: от 0 до 1\n\n"
            
            result_text += f"6. Комбинированная оценка: {combined_score:.3f}\n"
            result_text += f"   • Взвешенная оценка по всем метрикам\n"
            result_text += f"   • Диапазон: от 0 до 1\n\n"
            
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
            self.comparison_result.setText(f"Ошибка при сравнении: {str(e)}")
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
        
    def select_registration_file(self):
        """Выбор файла для регистрации"""
        if not self.username_input.text():
            self.show_status("Введите имя пользователя", "warning")
            return
            
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите аудио файл",
            "",
            "Audio Files (*.wav *.mp3 *.ogg *.flac);;All Files (*.*)"
        )
        
        if file_path:
            self.file_path_label.setText(f"Выбран файл: {os.path.basename(file_path)}")
            self.process_registration_file(file_path)
            
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
            
    def process_registration_file(self, file_path):
        """Обработка файла для регистрации"""
        self.processing_thread = FileProcessingThread(file_path)
        self.processing_thread.finished.connect(self.handle_registration_audio)
        self.processing_thread.error.connect(lambda msg: self.show_status(f"Ошибка: {msg}", "error"))
        self.processing_thread.start()
        self.show_status("Обработка файла...", "info")
        
    def process_verification_file(self, file_path):
        """Обработка файла для верификации"""
        self.processing_thread = FileProcessingThread(file_path)
        self.processing_thread.finished.connect(self.verify_voice)
        self.processing_thread.error.connect(lambda msg: self.verify_result.setText(f"Ошибка: {msg}"))
        self.processing_thread.start()
        self.verify_result.setText("Обработка файла...")
        
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
            
        except Exception as e:
            self.show_status(f"Ошибка при обработке файла: {str(e)}", "error")
            
    def verify_voice(self, audio_data):
        """Верификация записанного голоса"""
        try:
            # Извлечение эмбеддинга для проверяемого голоса
            with torch.no_grad():
                audio_tensor = torch.FloatTensor(audio_data).cuda()
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

def main():
    app = QApplication(sys.argv)
    window = VoiceAuthApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
