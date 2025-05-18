import os
import torch
import numpy as np
import soundfile as sf
from model import ECAPA_TDNN
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import silhouette_score
from datetime import datetime

class SpeakerClustering:
    def __init__(self, model_path, threshold=0.7, eps=0.3, min_samples=2, min_clusters=None, max_clusters=None):
        """
        Initialize the speaker clustering system
        Args:
            model_path: Path to the pretrained ECAPA-TDNN model
            threshold: Similarity threshold for considering two speakers as the same
            eps: The maximum distance between two samples for them to be considered neighbors in DBSCAN
            min_samples: The number of samples in a neighborhood for a point to be considered a core point
            min_clusters: Minimum number of clusters to aim for
            max_clusters: Maximum number of clusters to aim for
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ECAPA_TDNN(C=1024).to(self.device)
        
        # Load and process state dict
        state_dict = torch.load(model_path, map_location=self.device)
        processed_dict = {}
        
        # Remove 'speaker_encoder.' prefix from keys
        for key, value in state_dict.items():
            if key.startswith('speaker_encoder.'):
                new_key = key.replace('speaker_encoder.', '')
                processed_dict[new_key] = value
            elif not key.startswith('speaker_loss.'):  # Игнорируем веса классификатора
                processed_dict[key] = value
                
        self.model.load_state_dict(processed_dict)
        self.model.eval()
        self.threshold = threshold
        self.initial_eps = eps
        self.min_samples = min_samples
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters

    def extract_embedding(self, audio_path):
        """
        Extract speaker embedding from an audio file
        """
        # Read audio file
        audio, sr = sf.read(audio_path)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Resample to 16kHz if needed
        if sr != 16000:
            from scipy import signal
            audio = signal.resample(audio, int(len(audio) * 16000 / sr))
        
        # Convert to torch tensor
        audio = torch.FloatTensor(audio).unsqueeze(0).to(self.device)
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.model(audio, aug=False)
            embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding.cpu().numpy()

    def find_optimal_eps(self, embeddings, eps_range):
        """
        Find optimal eps parameter for DBSCAN
        """
        best_score = -1
        best_eps = self.initial_eps
        best_n_clusters = 0
        
        for eps in eps_range:
            clustering = DBSCAN(eps=eps, min_samples=self.min_samples, metric='cosine').fit(embeddings)
            labels = clustering.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            # Проверяем, соответствует ли количество кластеров заданным ограничениям
            if self.min_clusters and n_clusters < self.min_clusters:
                continue
            if self.max_clusters and n_clusters > self.max_clusters:
                continue
            
            # Вычисляем score только если есть больше одного кластера
            if n_clusters > 1:
                try:
                    score = silhouette_score(embeddings, labels, metric='cosine')
                    if score > best_score:
                        best_score = score
                        best_eps = eps
                        best_n_clusters = n_clusters
                except:
                    continue
        
        print(f"Optimal eps: {best_eps}, Number of clusters: {best_n_clusters}")
        return best_eps

    def cluster_speakers(self, audio_files):
        """
        Cluster audio files by speaker
        Args:
            audio_files: List of paths to audio files
        Returns:
            Dictionary mapping cluster IDs to lists of audio files
        """
        print("Extracting embeddings...")
        embeddings = []
        valid_files = []
        for audio_file in tqdm(audio_files):
            try:
                embedding = self.extract_embedding(audio_file)
                embeddings.append(embedding)
                valid_files.append(audio_file)
            except Exception as e:
                print(f"Error processing {audio_file}: {str(e)}")
                continue
        
        if not embeddings:
            return {}
        
        embeddings = np.vstack(embeddings)
        
        print("Finding optimal clustering parameters...")
        # Создаем диапазон значений eps для поиска
        eps_range = np.arange(0.1, 0.6, 0.02)
        optimal_eps = self.find_optimal_eps(embeddings, eps_range)
        
        print("Clustering speakers...")
        # Use DBSCAN with optimal parameters
        clustering = DBSCAN(eps=optimal_eps, min_samples=self.min_samples, metric='cosine').fit(embeddings)
        
        # Group files by cluster
        clusters = {}
        for idx, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(valid_files[idx])
        
        return clusters

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Cluster audio files by speaker')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained ECAPA-TDNN model')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing audio files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save clustered files')
    parser.add_argument('--threshold', type=float, default=0.7, help='Similarity threshold')
    parser.add_argument('--eps', type=float, default=0.3, help='Initial DBSCAN eps parameter')
    parser.add_argument('--min_samples', type=int, default=2, help='DBSCAN min_samples parameter')
    parser.add_argument('--min_clusters', type=int, default=None, help='Minimum number of clusters to aim for')
    parser.add_argument('--max_clusters', type=int, default=None, help='Maximum number of clusters to aim for')
    
    args = parser.parse_args()
    
    # Создаем уникальную директорию для этого запуска
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"clustering_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Сохраняем параметры запуска
    with open(os.path.join(run_dir, "params.txt"), "w") as f:
        f.write(f"Clustering parameters:\n")
        f.write(f"Input directory: {args.input_dir}\n")
        f.write(f"Model path: {args.model_path}\n")
        f.write(f"Threshold: {args.threshold}\n")
        f.write(f"Initial eps: {args.eps}\n")
        f.write(f"Min samples: {args.min_samples}\n")
        f.write(f"Min clusters: {args.min_clusters}\n")
        f.write(f"Max clusters: {args.max_clusters}\n")
    
    # Get list of audio files
    audio_files = []
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(('.wav', '.flac', '.mp3')):
                audio_files.append(os.path.join(root, file))
    
    if not audio_files:
        print("No audio files found!")
        return
    
    print(f"Results will be saved in: {run_dir}")
    
    # Initialize clustering
    clustering = SpeakerClustering(
        model_path=args.model_path,
        threshold=args.threshold,
        eps=args.eps,
        min_samples=args.min_samples,
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters
    )
    
    # Perform clustering
    clusters = clustering.cluster_speakers(audio_files)
    
    # Создаем файл с итоговой статистикой
    stats_file = os.path.join(run_dir, "clustering_stats.txt")
    with open(stats_file, "w") as f:
        f.write(f"Clustering Statistics:\n")
        f.write(f"Total files processed: {len(audio_files)}\n")
        f.write(f"Number of clusters found: {len(clusters)}\n")
        f.write("\nCluster sizes:\n")
        for cluster_id, files in clusters.items():
            cluster_name = "noise" if cluster_id == -1 else f"speaker_{cluster_id}"
            f.write(f"{cluster_name}: {len(files)} files\n")
    
    print("\nClustering results:")
    for cluster_id, files in clusters.items():
        cluster_name = "noise" if cluster_id == -1 else f"speaker_{cluster_id}"
        cluster_dir = os.path.join(run_dir, cluster_name)
        os.makedirs(cluster_dir, exist_ok=True)
        
        print(f"\n{cluster_name} ({len(files)} files):")
        for file in files:
            # Copy file to cluster directory
            import shutil
            filename = os.path.basename(file)
            shutil.copy2(file, os.path.join(cluster_dir, filename))
            print(f"  - {filename}")
    
    print(f"\nClustering complete! Results saved in: {run_dir}")
    print(f"Check {stats_file} for clustering statistics")

if __name__ == "__main__":
    main() 