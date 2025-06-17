import torch
import torchaudio
import numpy as np
from speechbrain.inference.speaker import SpeakerRecognition
import soundfile as sf
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

class SpeakerVerifier:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the speaker verification system using SpeechBrain's ECAPA-TDNN model
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        self.verification = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": device}
        )
        
    def verify_files(self, file1_path, file2_path):
        """
        Verify if two audio files are from the same speaker
        Args:
            file1_path: Path to the first audio file
            file2_path: Path to the second audio file
        Returns:
            tuple: (score, prediction, detailed_metrics)
            - score: Similarity score between 0 and 1
            - prediction: 1 if same speaker, 0 if different speakers
            - detailed_metrics: Dictionary containing detailed comparison metrics
        """
        # Get base verification score and prediction
        score, prediction = self.verification.verify_files(file1_path, file2_path)
        
        # Get detailed embeddings for additional metrics
        signal1, fs1 = torchaudio.load(file1_path)
        signal2, fs2 = torchaudio.load(file2_path)
        
        # Extract embeddings
        embedding1 = self.verification.encode_batch(signal1)
        embedding2 = self.verification.encode_batch(signal2)
        
        # Convert to numpy for additional metrics
        emb1 = embedding1.squeeze().cpu().numpy()
        emb2 = embedding2.squeeze().cpu().numpy()
        
        # Normalize embeddings
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)
        
        # Calculate additional metrics
        detailed_metrics = {
            'base_score': float(score),
            'cosine_similarity': 1 - cosine(emb1, emb2),
            'pearson_correlation': pearsonr(emb1, emb2)[0],
            'euclidean_distance': float(np.linalg.norm(emb1 - emb2)),
            'manhattan_distance': float(np.sum(np.abs(emb1 - emb2)))
        }
        
        return score, prediction, detailed_metrics
    
    def get_embedding(self, file_path):
        """
        Extract speaker embedding from an audio file
        Args:
            file_path: Path to the audio file
        Returns:
            numpy.ndarray: Speaker embedding vector
        """
        signal, fs = torchaudio.load(file_path)
        embedding = self.verification.encode_batch(signal)
        return embedding.squeeze().cpu().numpy()
    
    def compare_embeddings(self, embedding1, embedding2):
        """
        Compare two speaker embeddings
        Args:
            embedding1: First speaker embedding
            embedding2: Second speaker embedding
        Returns:
            tuple: (score, detailed_metrics)
        """
        # Normalize embeddings
        emb1 = embedding1 / np.linalg.norm(embedding1)
        emb2 = embedding2 / np.linalg.norm(embedding2)
        
        # Calculate metrics
        detailed_metrics = {
            'cosine_similarity': 1 - cosine(emb1, emb2),
            'pearson_correlation': pearsonr(emb1, emb2)[0],
            'euclidean_distance': float(np.linalg.norm(emb1 - emb2)),
            'manhattan_distance': float(np.sum(np.abs(emb1 - emb2)))
        }
        
        # Calculate combined score
        weights = {
            'cosine': 0.4,
            'pearson': 0.3,
            'euclidean': 0.2,
            'manhattan': 0.1
        }
        
        # Normalize distances to similarities
        euclidean_sim = np.exp(-detailed_metrics['euclidean_distance'])
        manhattan_sim = np.exp(-detailed_metrics['manhattan_distance'] / len(emb1))
        
        combined_score = (
            weights['cosine'] * detailed_metrics['cosine_similarity'] +
            weights['pearson'] * detailed_metrics['pearson_correlation'] +
            weights['euclidean'] * euclidean_sim +
            weights['manhattan'] * manhattan_sim
        )
        
        return combined_score, detailed_metrics 