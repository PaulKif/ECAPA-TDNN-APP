import sys
import os
import torch
import sounddevice as sd
import soundfile as sf
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, 
                           QVBoxLayout, QWidget, QLabel, QLineEdit)
from PyQt5.QtCore import Qt
from model import ECAPA_TDNN
from sklearn.cluster import KMeans

class VoiceAuthApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice Authentication System")
        self.setGeometry(100, 100, 400, 300)
        
        # Initialize the model
        self.model = ECAPA_TDNN(C=1024).cuda()
        self.model.eval()
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        
        # Create UI elements
        self.status_label = QLabel("Status: Ready")
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Enter username")
        
        self.record_button = QPushButton("Record Voice")
        self.record_button.clicked.connect(self.record_voice)
        
        self.verify_button = QPushButton("Verify Voice")
        self.verify_button.clicked.connect(self.verify_voice)
        
        # Add widgets to layout
        layout.addWidget(self.status_label)
        layout.addWidget(self.username_input)
        layout.addWidget(self.record_button)
        layout.addWidget(self.verify_button)
        
        main_widget.setLayout(layout)
        
        # Create directory for voice embeddings if it doesn't exist
        if not os.path.exists("voice_embeddings"):
            os.makedirs("voice_embeddings")
    
    def record_voice(self):
        self.status_label.setText("Recording... Speak now!")
        QApplication.processEvents()
        
        # Record audio
        duration = 3  # seconds
        sample_rate = 16000
        recording = sd.rec(int(duration * sample_rate), 
                         samplerate=sample_rate, 
                         channels=1, 
                         dtype='float32')
        sd.wait()
        
        # Save recording
        username = self.username_input.text()
        if not username:
            self.status_label.setText("Error: Please enter a username")
            return
            
        # Save audio file
        audio_path = f"voice_embeddings/{username}.wav"
        sf.write(audio_path, recording, sample_rate)
        
        # Extract and save embedding
        with torch.no_grad():
            # Reshape audio tensor to match model's expected input
            audio_tensor = torch.FloatTensor(recording).cuda()
            audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
            audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dimension
            embedding = self.model(audio_tensor, aug=False)
            embedding = embedding.cpu().numpy()
            
        # Save embedding
        np.save(f"voice_embeddings/{username}.npy", embedding)
        
        self.status_label.setText(f"Voice recorded and saved for {username}")
    
    def verify_voice(self):
        self.status_label.setText("Recording for verification...")
        QApplication.processEvents()
        
        # Record audio
        duration = 3  # seconds
        sample_rate = 16000
        recording = sd.rec(int(duration * sample_rate), 
                         samplerate=sample_rate, 
                         channels=1, 
                         dtype='float32')
        sd.wait()
        
        # Extract embedding
        with torch.no_grad():
            # Reshape audio tensor to match model's expected input
            audio_tensor = torch.FloatTensor(recording).cuda()
            audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
            audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dimension
            test_embedding = self.model(audio_tensor, aug=False)
            test_embedding = test_embedding.cpu().numpy()
        
        # Compare with saved embeddings
        best_match = None
        best_score = -1
        
        for file in os.listdir("voice_embeddings"):
            if file.endswith(".npy"):
                saved_embedding = np.load(f"voice_embeddings/{file}")
                score = np.dot(test_embedding, saved_embedding.T)
                if score > best_score:
                    best_score = score
                    best_match = file[:-4]  # Remove .npy extension
        
        if best_score > 0.7:  # Threshold for matching
            self.status_label.setText(f"Verified as: {best_match}")
        else:
            self.status_label.setText("No match found")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VoiceAuthApp()
    window.show()
    sys.exit(app.exec_()) 