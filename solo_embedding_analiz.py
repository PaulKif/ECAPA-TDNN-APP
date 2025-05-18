import torch
from model import ECAPA_TDNN
import soundfile as sf
import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def process_audio_file(audio_path, target_length=48000):
    audio, _ = sf.read(audio_path)
    print("Original audio shape:", audio.shape)
    
    # Обрезка или дополнение до целевой длины
    #if len(audio) > target_length:
    #    audio = audio[:target_length]
    #else:
    #    # Дополнение нулями
    #    audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
    
    print("Processed audio shape:", audio.shape)
    return audio

# Загрузка модели
model = ECAPA_TDNN(C=1024).cuda()
model.eval()

# Путь к вашему аудиофайлу
audio_path = 'ru/wav/common_voice_ru_41910911.wav'

# Обработка аудиофайла
audio = process_audio_file(audio_path)

# Преобразование в тензор и изменение размерностей
audio_tensor = torch.FloatTensor(audio).cuda()
print("Tensor shape after FloatTensor:", audio_tensor.shape)

# Добавляем только batch dimension: [time_steps] -> [batch_size, time_steps]
audio_tensor = audio_tensor.unsqueeze(0)  # [1, time_steps]
print("Final tensor shape:", audio_tensor.shape)

# Извлечение эмбеддингов
with torch.no_grad():
    embedding = model(audio_tensor, aug=False)
    embedding = embedding.cpu().numpy()

print("Embedding shape:", embedding.shape)

# 1. Базовый анализ
print("\n1. Базовый анализ эмбеддинга:")
print("Размерность эмбеддинга:", embedding.shape)
print("Минимальное значение:", np.min(embedding))
print("Максимальное значение:", np.max(embedding))
print("Среднее значение:", np.mean(embedding))
print("Стандартное отклонение:", np.std(embedding))

# 2. Визуализация распределения значений
plt.figure(figsize=(10, 5))
plt.hist(embedding.flatten(), bins=50)
plt.title('Распределение значений в эмбеддинге')
plt.xlabel('Значение')
plt.ylabel('Частота')
plt.show()

# 3. Визуализация эмбеддинга как тепловой карты
plt.figure(figsize=(10, 2))
plt.imshow(embedding, aspect='auto', cmap='viridis')
plt.colorbar()
plt.title('Тепловая карта эмбеддинга')
plt.show()

# 4. PCA для уменьшения размерности и визуализации
pca = PCA(n_components=2)
embedding_2d = pca.fit_transform(embedding)
plt.figure(figsize=(8, 8))
plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1])
plt.title('PCA визуализация эмбеддинга')
plt.xlabel('Первая главная компонента')
plt.ylabel('Вторая главная компонента')
plt.show()

# 5. t-SNE для визуализации (если у вас несколько эмбеддингов)
# tsne = TSNE(n_components=2)
# embedding_tsne = tsne.fit_transform(embedding)
# plt.figure(figsize=(8, 8))
# plt.scatter(embedding_tsne[:, 0], embedding_tsne[:, 1])
# plt.title('t-SNE визуализация эмбеддинга')
# plt.show()

# 6. Сохранение эмбеддинга для дальнейшего использования
np.save('embedding.npy', embedding)
print("\nЭмбеддинг сохранен в файл 'embedding.npy'")
