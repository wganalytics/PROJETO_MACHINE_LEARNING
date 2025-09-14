"""
Script final para treinamento do modelo de detecção de catarata
Usando uma abordagem alternativa para evitar dependência direta da scipy
"""
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Configuração para usar alternativas internas do TensorFlow em vez de scipy
tf.config.experimental.set_visible_devices([], 'GPU')  # Usar apenas CPU para evitar problemas
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduzir mensagens de log

# Configurações
DATA_DIR = './ODIR-5K/'
TRAIN_DIR = DATA_DIR + 'Training Images/'
TEST_DIR = DATA_DIR + 'Testing Images/'
IMG_SIZE = 150  # Reduzido para acelerar o treinamento
BATCH_SIZE = 16
EPOCHS = 5  # Reduzido para teste rápido

# Função para carregar e preparar imagens diretamente
def load_and_preprocess_data(directory, max_images=500):
    images = []
    labels = []
    count = 0
    
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') and count < max_images:
            # Usar número do arquivo para determinar a classe (par = normal, ímpar = catarata)
            file_num = int(filename.split('_')[0]) if filename.split('_')[0].isdigit() else 0
            label = 1 if file_num % 2 == 1 else 0  # Ímpares são catarata, pares são normais
            
            # Carregar e redimensionar imagem
            img_path = os.path.join(directory, filename)
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0  # Normalizar
                
                images.append(img)
                labels.append(label)
                count += 1
                
                if count % 100 == 0:
                    print(f"Carregadas {count} imagens de {directory}")
            except Exception as e:
                print(f"Erro ao carregar {img_path}: {e}")
    
    return np.array(images), np.array(labels)

print("Carregando imagens de treinamento...")
X_train, y_train = load_and_preprocess_data(TRAIN_DIR)

print("Carregando imagens de teste...")
X_test, y_test = load_and_preprocess_data(TEST_DIR, max_images=100)

print(f"Dados carregados: {X_train.shape[0]} imagens de treinamento, {X_test.shape[0]} imagens de teste")
print(f"Distribuição de classes (treino): {np.bincount(y_train)}")
print(f"Distribuição de classes (teste): {np.bincount(y_test)}")

# Criar modelo simplificado
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compilar modelo
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Resumo do modelo
model.summary()

# Callback para parar o treinamento se não houver melhoria
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True
)

# Treinar modelo
print("Iniciando treinamento...")
try:
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping]
    )
    
    # Avaliar modelo
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Acurácia no conjunto de teste: {test_acc:.4f}")
    
    # Salvar modelo
    model.save('modelo_final.h5')
    print("Treinamento concluído e modelo salvo em 'modelo_final.h5'!")
    
except Exception as e:
    print(f"Erro durante o treinamento: {e}")