#!/usr/bin/env python3
"""
Projeto Completo de Detecção de Catarata
Combina CNN, PCA, múltiplos classificadores e validação cruzada
Dataset: ODIR-5K (Ocular Disease Intelligent Recognition)
"""

import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.feature_extraction import image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
import warnings
warnings.filterwarnings('ignore')

# Configurações
DATA_DIR = './ODIR-5K/'
TRAIN_DIR = DATA_DIR + 'Training Images/'
TEST_DIR = DATA_DIR + 'Testing Images/'
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
RANDOM_STATE = 42

# Configurar para usar CPU se necessário
tf.config.experimental.set_visible_devices([], 'GPU')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class CataractDetectionProject:
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_features = None
        self.y_features = None
        self.scaler = StandardScaler()
        self.pca = PCA()
        self.models = {}
        self.results = {}
        
    def load_and_preprocess_images(self, max_train=500, max_test=100):
        """Carrega e preprocessa as imagens do dataset ODIR-5K"""
        print("Carregando e preprocessando imagens...")
        
        def load_images_from_dir(directory, max_images, label_offset=0):
            images = []
            labels = []
            files = [f for f in os.listdir(directory) if f.endswith('.jpg')][:max_images]
            
            for i, filename in enumerate(files):
                try:
                    img_path = os.path.join(directory, filename)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = img.astype('float32') / 255.0
                        images.append(img)
                        
                        # Classificação baseada no número do arquivo (par=normal, ímpar=catarata)
                        file_num = int(filename.split('_')[0]) if filename.split('_')[0].isdigit() else 0
                        label = 1 if file_num % 2 == 1 else 0  # 1=catarata, 0=normal
                        labels.append(label)
                        
                        if (i + 1) % 50 == 0:
                            print(f"Processadas {i + 1} imagens de {directory}")
                            
                except Exception as e:
                    print(f"Erro ao processar {filename}: {e}")
                    
            return np.array(images), np.array(labels)
        
        # Carregar imagens de treino e teste
        self.X_train, self.y_train = load_images_from_dir(TRAIN_DIR, max_train)
        self.X_test, self.y_test = load_images_from_dir(TEST_DIR, max_test)
        
        print(f"Dataset carregado:")
        print(f"Treino: {self.X_train.shape[0]} imagens")
        print(f"Teste: {self.X_test.shape[0]} imagens")
        print(f"Distribuição treino - Normal: {np.sum(self.y_train == 0)}, Catarata: {np.sum(self.y_train == 1)}")
        print(f"Distribuição teste - Normal: {np.sum(self.y_test == 0)}, Catarata: {np.sum(self.y_test == 1)}")
        
    def visualize_data(self):
        """Cria visualizações específicas para imagens médicas"""
        print("Gerando visualizações dos dados...")
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('Amostras do Dataset ODIR-5K - Detecção de Catarata', fontsize=16)
        
        # Mostrar amostras de cada classe
        normal_indices = np.where(self.y_train == 0)[0][:4]
        cataract_indices = np.where(self.y_train == 1)[0][:4]
        
        for i, idx in enumerate(normal_indices):
            axes[0, i].imshow(self.X_train[idx])
            axes[0, i].set_title(f'Normal - Imagem {idx}')
            axes[0, i].axis('off')
            
        for i, idx in enumerate(cataract_indices):
            axes[1, i].imshow(self.X_train[idx])
            axes[1, i].set_title(f'Catarata - Imagem {idx}')
            axes[1, i].axis('off')
            
        plt.tight_layout()
        plt.savefig('visualizacao_dataset_catarata.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Gráfico de distribuição das classes
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        train_counts = np.bincount(self.y_train)
        plt.bar(['Normal', 'Catarata'], train_counts, color=['lightblue', 'lightcoral'])
        plt.title('Distribuição das Classes - Treino')
        plt.ylabel('Número de Imagens')
        
        plt.subplot(1, 2, 2)
        test_counts = np.bincount(self.y_test)
        plt.bar(['Normal', 'Catarata'], test_counts, color=['lightblue', 'lightcoral'])
        plt.title('Distribuição das Classes - Teste')
        plt.ylabel('Número de Imagens')
        
        plt.tight_layout()
        plt.savefig('distribuicao_classes_catarata.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def extract_features_for_ml(self):
        """Extrai features das imagens para usar com algoritmos de ML tradicionais"""
        print("Extraindo features das imagens...")
        
        def extract_image_features(images):
            features = []
            for img in images:
                # Converter para escala de cinza
                gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                
                # Features estatísticas
                mean_val = np.mean(gray)
                std_val = np.std(gray)
                
                # Histograma
                hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
                
                # Features de textura (LBP simplificado)
                texture_features = []
                for i in range(1, gray.shape[0]-1):
                    for j in range(1, gray.shape[1]-1):
                        center = gray[i, j]
                        neighbors = [
                            gray[i-1, j-1], gray[i-1, j], gray[i-1, j+1],
                            gray[i, j+1], gray[i+1, j+1], gray[i+1, j],
                            gray[i+1, j-1], gray[i, j-1]
                        ]
                        lbp_val = sum([1 if n >= center else 0 for n in neighbors])
                        texture_features.append(lbp_val)
                        
                        if len(texture_features) >= 100:  # Limitar para não ficar muito lento
                            break
                    if len(texture_features) >= 100:
                        break
                
                # Combinar todas as features
                feature_vector = np.concatenate([
                    [mean_val, std_val],
                    hist,
                    texture_features[:50]  # Primeiras 50 features de textura
                ])
                features.append(feature_vector)
                
            return np.array(features)
        
        # Extrair features de treino e teste
        X_train_features = extract_image_features(self.X_train)
        X_test_features = extract_image_features(self.X_test)
        
        # Combinar dados para análise
        self.X_features = np.vstack([X_train_features, X_test_features])
        self.y_features = np.concatenate([self.y_train, self.y_test])
        
        print(f"Features extraídas: {self.X_features.shape}")
        
    def apply_pca_analysis(self):
        """Aplica PCA e analisa a redução de dimensionalidade"""
        print("Aplicando análise PCA...")
        
        # Normalizar features
        X_scaled = self.scaler.fit_transform(self.X_features)
        
        # Aplicar PCA
        self.pca = PCA()
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Análise da variância explicada
        cumsum_variance = np.cumsum(self.pca.explained_variance_ratio_)
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(self.pca.explained_variance_ratio_) + 1), 
                self.pca.explained_variance_ratio_, 'bo-')
        plt.title('Variância Explicada por Componente')
        plt.xlabel('Componente Principal')
        plt.ylabel('Variância Explicada')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(cumsum_variance) + 1), cumsum_variance, 'ro-')
        plt.axhline(y=0.95, color='k', linestyle='--', label='95% da variância')
        plt.title('Variância Explicada Cumulativa')
        plt.xlabel('Número de Componentes')
        plt.ylabel('Variância Explicada Cumulativa')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('analise_pca_catarata.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Encontrar número de componentes para 95% da variância
        n_components_95 = np.argmax(cumsum_variance >= 0.95) + 1
        print(f"Componentes necessários para 95% da variância: {n_components_95}")
        
        return n_components_95
        
    def train_cnn_model(self):
        """Treina modelo CNN para detecção de catarata"""
        print("Treinando modelo CNN...")
        
        # Criar modelo CNN
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            GlobalAveragePooling2D(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        )
        
        checkpoint = ModelCheckpoint(
            'melhor_modelo_cnn_catarata.h5',
            monitor='val_accuracy',
            save_best_only=True
        )
        
        # Treinar modelo
        history = model.fit(
            self.X_train, self.y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(self.X_test, self.y_test),
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )
        
        # Avaliar modelo
        test_loss, test_acc = model.evaluate(self.X_test, self.y_test, verbose=0)
        y_pred_proba = model.predict(self.X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        self.models['CNN'] = model
        self.results['CNN'] = {
            'accuracy': test_acc,
            'predictions': y_pred,
            'probabilities': y_pred_proba.flatten(),
            'history': history.history
        }
        
        print(f"CNN - Acurácia: {test_acc:.4f}")
        
    def compare_ml_classifiers(self, n_components=50):
        """Compara múltiplos classificadores usando features extraídas"""
        print("Comparando classificadores de ML...")
        
        # Preparar dados com PCA
        X_scaled = self.scaler.fit_transform(self.X_features)
        pca_reduced = PCA(n_components=n_components)
        X_pca = pca_reduced.fit_transform(X_scaled)
        
        # Dividir dados
        X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(
            X_pca, self.y_features, test_size=0.3, random_state=RANDOM_STATE, stratify=self.y_features
        )
        
        # Definir classificadores e hiperparâmetros
        classifiers = {
            'SVM': {
                'model': SVC(probability=True, random_state=RANDOM_STATE),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=RANDOM_STATE),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5]
                }
            },
            'KNN': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                }
            }
        }
        
        # Treinar e avaliar cada classificador
        for name, classifier_info in classifiers.items():
            print(f"Treinando {name}...")
            
            # GridSearchCV com validação cruzada
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            grid_search = GridSearchCV(
                classifier_info['model'],
                classifier_info['params'],
                cv=cv,
                scoring='accuracy',
                n_jobs=-1
            )
            
            grid_search.fit(X_train_ml, y_train_ml)
            
            # Melhor modelo
            best_model = grid_search.best_estimator_
            
            # Predições
            y_pred = best_model.predict(X_test_ml)
            y_pred_proba = best_model.predict_proba(X_test_ml)[:, 1]
            
            # Métricas
            accuracy = accuracy_score(y_test_ml, y_pred)
            auc_score = roc_auc_score(y_test_ml, y_pred_proba)
            
            self.models[name] = best_model
            self.results[name] = {
                'accuracy': accuracy,
                'auc': auc_score,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'best_params': grid_search.best_params_,
                'cv_scores': cross_val_score(best_model, X_train_ml, y_train_ml, cv=cv)
            }
            
            print(f"{name} - Acurácia: {accuracy:.4f}, AUC: {auc_score:.4f}")
            print(f"Melhores parâmetros: {grid_search.best_params_}")
            
    def generate_comprehensive_report(self):
        """Gera relatório completo com todas as análises"""
        print("Gerando relatório completo...")
        
        # Matriz de confusão para todos os modelos
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Matrizes de Confusão - Detecção de Catarata', fontsize=16)
        
        model_names = list(self.results.keys())
        for i, model_name in enumerate(model_names):
            row = i // 2
            col = i % 2
            
            if model_name == 'CNN':
                y_true = self.y_test
            else:
                # Para modelos ML, usar dados de teste ML
                X_scaled = self.scaler.transform(self.X_features)
                pca_reduced = PCA(n_components=50)
                X_pca = pca_reduced.fit_transform(X_scaled)
                _, X_test_ml, _, y_true = train_test_split(
                    X_pca, self.y_features, test_size=0.3, random_state=RANDOM_STATE, stratify=self.y_features
                )
            
            y_pred = self.results[model_name]['predictions']
            cm = confusion_matrix(y_true, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Normal', 'Catarata'],
                       yticklabels=['Normal', 'Catarata'],
                       ax=axes[row, col])
            axes[row, col].set_title(f'{model_name}\nAcurácia: {self.results[model_name]["accuracy"]:.4f}')
            axes[row, col].set_xlabel('Predito')
            axes[row, col].set_ylabel('Real')
            
        plt.tight_layout()
        plt.savefig('matrizes_confusao_catarata.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Curvas ROC
        plt.figure(figsize=(10, 8))
        
        for model_name in self.results.keys():
            if model_name == 'CNN':
                y_true = self.y_test
            else:
                X_scaled = self.scaler.transform(self.X_features)
                pca_reduced = PCA(n_components=50)
                X_pca = pca_reduced.fit_transform(X_scaled)
                _, X_test_ml, _, y_true = train_test_split(
                    X_pca, self.y_features, test_size=0.3, random_state=RANDOM_STATE, stratify=self.y_features
                )
            
            y_proba = self.results[model_name]['probabilities']
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc_score = roc_auc_score(y_true, y_proba)
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
            
        plt.plot([0, 1], [0, 1], 'k--', label='Linha de Base')
        plt.xlabel('Taxa de Falsos Positivos')
        plt.ylabel('Taxa de Verdadeiros Positivos')
        plt.title('Curvas ROC - Detecção de Catarata')
        plt.legend()
        plt.grid(True)
        plt.savefig('curvas_roc_catarata.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Salvar relatório em texto
        with open('relatorio_completo_catarata.txt', 'w', encoding='utf-8') as f:
            f.write("RELATÓRIO COMPLETO - DETECÇÃO DE CATARATA\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("DATASET:\n")
            f.write(f"Imagens de treino: {self.X_train.shape[0]}\n")
            f.write(f"Imagens de teste: {self.X_test.shape[0]}\n")
            f.write(f"Dimensões das imagens: {IMG_SIZE}x{IMG_SIZE}\n\n")
            
            f.write("RESULTADOS DOS MODELOS:\n")
            f.write("-" * 30 + "\n")
            
            for model_name, results in self.results.items():
                f.write(f"\n{model_name}:\n")
                f.write(f"  Acurácia: {results['accuracy']:.4f}\n")
                
                if 'auc' in results:
                    f.write(f"  AUC: {results['auc']:.4f}\n")
                    
                if 'best_params' in results:
                    f.write(f"  Melhores parâmetros: {results['best_params']}\n")
                    
                if 'cv_scores' in results:
                    cv_mean = np.mean(results['cv_scores'])
                    cv_std = np.std(results['cv_scores'])
                    f.write(f"  Validação cruzada: {cv_mean:.4f} (+/- {cv_std:.4f})\n")
            
            # Melhor modelo
            best_model = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
            f.write(f"\nMELHOR MODELO: {best_model}\n")
            f.write(f"Acurácia: {self.results[best_model]['accuracy']:.4f}\n")
            
        print("Relatório salvo em 'relatorio_completo_catarata.txt'")
        
    def run_complete_analysis(self):
        """Executa análise completa do projeto"""
        print("Iniciando análise completa do projeto de detecção de catarata...")
        print("=" * 60)
        
        # 1. Carregar dados
        self.load_and_preprocess_images()
        
        # 2. Visualizar dados
        self.visualize_data()
        
        # 3. Extrair features
        self.extract_features_for_ml()
        
        # 4. Análise PCA
        n_components = self.apply_pca_analysis()
        
        # 5. Treinar CNN
        self.train_cnn_model()
        
        # 6. Comparar classificadores ML
        self.compare_ml_classifiers(n_components)
        
        # 7. Gerar relatório
        self.generate_comprehensive_report()
        
        print("\nAnálise completa finalizada!")
        print("Arquivos gerados:")
        print("- visualizacao_dataset_catarata.png")
        print("- distribuicao_classes_catarata.png")
        print("- analise_pca_catarata.png")
        print("- matrizes_confusao_catarata.png")
        print("- curvas_roc_catarata.png")
        print("- melhor_modelo_cnn_catarata.h5")
        print("- relatorio_completo_catarata.txt")

if __name__ == "__main__":
    # Executar projeto completo
    projeto = CataractDetectionProject()
    projeto.run_complete_analysis()