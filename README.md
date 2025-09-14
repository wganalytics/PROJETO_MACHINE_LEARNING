# Projeto Completo de Detecção de Catarata

## Descrição
Este projeto implementa um sistema completo de detecção de catarata usando o dataset ODIR-5K (Ocular Disease Intelligent Recognition). O projeto combina técnicas avançadas de Machine Learning, incluindo Redes Neurais Convolucionais (CNN), análise de componentes principais (PCA), e múltiplos classificadores tradicionais.

## 📁 Estrutura do Projeto

```
projeto-catarata/
├── ODIR-5K/                    # Dataset ODIR-5K
│   ├── Training Images/        # Imagens de treinamento (7.000 imagens)
│   ├── Testing Images/         # Imagens de teste (1.000 imagens)
│   └── data.xlsx              # Metadados do dataset
├── projeto_completo_catarata.py # Script principal com análise completa
├── projeto_ml_otimizado.py     # Script otimizado para produção
├── treinar_final.py           # Script de treinamento CNN simples
├── Projeto_Final_AM.pdf       # Documentação do projeto
├── README.md                  # Este arquivo
├── requirements.txt           # Dependências do projeto
└── venv/                      # Ambiente virtual Python
```

## ⚙️ Funcionalidades Implementadas

### 1. Processamento de Imagens
- Carregamento e redimensionamento de imagens para 224x224 pixels
- Normalização de pixels (0-1)
- Conversão de espaço de cores BGR para RGB
- Classificação automática baseada no nome do arquivo (par=normal, ímpar=catarata)

### 2. Visualizações de Dados
- Amostras representativas de cada classe (normal vs catarata)
- Distribuição das classes no dataset
- Análise da variância explicada pelo PCA
- Matrizes de confusão para todos os modelos
- Curvas ROC para comparação de performance

### 3. Extração de Features
- Features estatísticas (média, desvio padrão)
- Histogramas de intensidade
- Features de textura (Local Binary Pattern simplificado)
- Redução de dimensionalidade com PCA

### 4. Modelos Implementados

#### Rede Neural Convolucional (CNN)
- Arquitetura com 3 camadas convolucionais
- MaxPooling e Dropout para regularização
- GlobalAveragePooling para reduzir parâmetros
- Early stopping e checkpoint para melhor modelo

#### Classificadores Tradicionais
- **SVM (Support Vector Machine)**
  - Kernels: RBF e Linear
  - Otimização de hiperparâmetros com GridSearchCV
  
- **Random Forest**
  - Múltiplas árvores de decisão
  - Otimização do número de estimadores e profundidade
  
- **K-Nearest Neighbors (KNN)**
  - Diferentes valores de K
  - Métricas de distância: Euclidiana e Manhattan

### 5. Validação e Métricas
- Validação cruzada estratificada (5-fold)
- Métricas de avaliação:
  - Acurácia
  - AUC (Area Under Curve)
  - Matriz de confusão
  - Relatório de classificação
  - Curvas ROC

## 💻 Requisitos do Sistema

### Dependências Python
```bash
pip install -r requirements.txt
```

Ou instale manualmente:
```bash
pip install numpy>=1.21.0 pandas>=1.3.0 opencv-python>=4.5.0
pip install scikit-learn>=1.0.0 matplotlib>=3.4.0 seaborn>=0.11.0
pip install tensorflow>=2.8.0 tqdm>=4.62.0
```

### Requisitos de Hardware
- RAM: Mínimo 8GB (recomendado 16GB)
- Processador: Multi-core (o projeto usa paralelização)
- Espaço em disco: ~2GB para o dataset ODIR-5K

## 🚀 Como Executar

### Pré-requisitos
- Python 3.8+
- Bibliotecas listadas em `requirements.txt`
- Dataset ODIR-5K (deve estar na pasta `ODIR-5K/`)

### Passos de Execução

1. **Clone o repositório e navegue até o diretório:**
   ```bash
   cd projeto-catarata
   ```

2. **Crie e ative um ambiente virtual:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Execute o projeto completo:**
   ```bash
   python projeto_completo_catarata.py
   ```

5. **Ou execute o treinamento CNN simples:**
   ```bash
   python treinar_final.py
   ```

## 📊 Arquivos Gerados

Após a execução, os seguintes arquivos são criados:

### projeto_completo_catarata.py:
- `cataract_detection_results.png` - Visualizações dos dados
- `pca_analysis.png` - Análise de componentes principais
- `cnn_training_history.png` - Histórico de treinamento da CNN
- `ml_classifiers_comparison.png` - Comparação de classificadores
- `comprehensive_report.txt` - Relatório detalhado dos resultados
- `best_cnn_model.h5` - Melhor modelo CNN treinado
- `feature_data.pkl` - Features extraídas para ML tradicional

### treinar_final.py:
- `modelo_catarata.h5` - Modelo CNN treinado
- `historico_treinamento.png` - Gráficos de acurácia e perda

## 📈 Resultados Esperados

O projeto foi projetado para:
- Processar até 500 imagens de treino e 100 de teste (configurável)
- Atingir acurácia superior a 70% na detecção de catarata
- Comparar performance entre diferentes algoritmos de ML
- Identificar o melhor modelo para o problema específico
- Gerar relatórios detalhados com métricas de avaliação

## 🔧 Características Técnicas

### Otimizações Implementadas
- Uso de CPU apenas (evita problemas de compatibilidade GPU)
- Processamento em lotes para eficiência de memória
- Paralelização em algoritmos que suportam (n_jobs=-1)
- Early stopping para evitar overfitting
- Validação cruzada para resultados mais robustos

### Tratamento de Dados
- Classificação automática baseada em padrões do nome do arquivo
- Balanceamento natural das classes
- Normalização de features para algoritmos sensíveis à escala
- Redução de dimensionalidade para melhor performance

## ⚠️ Limitações e Considerações

1. **Dataset**: A classificação é baseada no padrão do nome do arquivo (par/ímpar)
2. **Processamento**: Limitado a subconjunto do dataset para viabilidade computacional
3. **Validação**: Seria ideal ter validação médica especializada dos resultados
4. **Hardware**: Otimizado para execução em CPU (compatibilidade Mac M1/M2)

## 🚀 Extensões Futuras

- Implementação de transfer learning com modelos pré-treinados
- Aumento de dados (data augmentation) mais sofisticado
- Ensemble de modelos para melhor performance
- Interface gráfica para uso clínico
- Validação com dataset médico real anotado por especialistas

## 👨‍💻 Autor

Projeto desenvolvido para a disciplina de Machine Learning - Pós-graduação UFG

## 📄 Licença

Este projeto é para fins educacionais e de pesquisa.