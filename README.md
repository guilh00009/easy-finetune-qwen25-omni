# AITrainer - Fine-tuning Multimodal Models while Preserving All Capabilities
#NEW FIXED VERSION MADE BY guilh00009
#Originally made by ZaccariaDev
[English](#english) | [Français](#français) | [Italiano](#italiano)

---

## English

### Overview

**AITrainer** is a flexible fine-tuning framework for **multimodal language models** that **preserves their complete capabilities** (audio, vision, video, text). Originally designed for Qwen2.5-Omni-3B, the custom wrapper approach can be adapted to other multimodal architectures that don't work with standard fine-tuning tools.

**Primary Use Case:** Qwen2.5-Omni-3B (tested and working)
**Adaptable to:** Other multimodal models with complex architectures or non-standard forward() implementations

The main challenge: Standard fine-tuning tools like `SFTTrainer` don't work with certain multimodal architectures. This project provides a working solution using custom wrappers and targeted LoRA application.

### 🌟 Key Features

- ✅ **Preserves multimodal capabilities** (audio, vision, video, text)
- ✅ **Adaptable custom wrapper approach** for complex model architectures
- ✅ **QLoRA fine-tuning** with 4-bit quantization (~7.5GB VRAM)
- ✅ **Efficient training** - only 0.87% of parameters trained (41M out of 4.7B on Qwen)
- ✅ **Docker environment** with CUDA 12.3.2 and PyTorch 2.6
- ✅ **FastAPI inference server** with LoRA adapter loading
- ✅ **Works with safetensors** format for model weights

### 📁 Project Structure

```
AITrainer/
├── Dockerfile                      # CUDA + PyTorch 2.6 + Transformers environment
├── docker-compose.yml              # Docker Compose configuration
├── experiments/
│   └── qwen25_omni_custom/         # Custom wrapper approach
│       ├── finetune_omni_wrapper.py  # Main training script
│       ├── inspect_model.py          # Model architecture inspector
│       └── test_finetuned_model.py   # Testing script
├── ai_server/
│   ├── main.py                     # FastAPI inference server
│   └── requirements.txt            # Python dependencies
├── data/
│   └── example_dataset.jsonl       # Example training dataset (JSONL format)
└── models/                         # Directory for LoRA adapters
```

### 🔧 How It Works

#### The Problem

Many multimodal models have complex architectures that break standard fine-tuning tools:
- **Example: Qwen2.5-Omni** has `Thinker` (language model) + `Talker` (speech) + multimodal encoders
- **Common issue:** `forward()` method not properly exposed or implemented
- **Result:** Standard `SFTTrainer` and similar tools fail

This framework solves these issues with a flexible approach that can be adapted to different architectures.

#### The Solution

1. **Custom Wrapper**: Create a `Qwen2_5OmniWrapper` class that:
   - Accesses the internal `thinker` component
   - Exposes a proper `forward()` method for text-only training
   - Delegates `generate()` to the full Omni model (preserves multimodal)

2. **LoRA Before Wrapping** (crucial):
   ```python
   # 1. Extract Thinker
   thinker = omni_model.thinker

   # 2. Apply LoRA to Thinker
   thinker_with_lora = get_peft_model(thinker, lora_config)

   # 3. Replace Thinker in omni_model
   omni_model.thinker = thinker_with_lora

   # 4. THEN create wrapper
   model = Qwen2_5OmniWrapper(omni_model)
   ```

3. **QLoRA**: 4-bit quantization + LoRA adapters to fit in ~7.5GB VRAM

### 🚀 Quick Start

#### Prerequisites

- Docker with NVIDIA runtime
- NVIDIA GPU with CUDA support (tested on RTX 3080)
- At least 16GB RAM and 10GB VRAM

#### 1. Clone and Setup

```bash
[git clone https://github.com/yourusername/AITrainer.git](https://github.com/guilh00009/easy-finetune-qwen25-omni.git)
cd AITrainer
```

#### 2. Download Base Model

Download Qwen2.5-Omni-3B from Hugging Face:
```bash
# Create a directory for the base model
mkdir -p base_model

# Download using git-lfs or huggingface-cli
huggingface-cli download Qwen/Qwen2.5-Omni-3B --local-dir base_model
```

#### 3. Prepare Your Dataset

Create a JSONL file in `data/` with your training examples. Supported formats:

**Simple chat:**
```json
{
  "type": "chat",
  "messages": [
    {"role": "user", "content": "What is X?"},
    {"role": "assistant", "content": "X is..."}
  ]
}
```

**Analysis with context:**
```json
{
  "type": "analysis",
  "context": {"today_log": {"day_of_flower": 18}},
  "messages": [
    {"role": "user", "content": "Analyze this"},
    {"role": "assistant", "content": "Analysis..."}
  ]
}
```

**Analysis with JSON payload:**
```json
{
  "type": "analysis_json",
  "context": {
    "session_config": {...},
    "products_catalog": {...},
    "today_log": {...}
  },
  "messages": [
    {"role": "user", "content": "Analyze this payload"},
    {"role": "assistant", "content": "Complete analysis..."}
  ]
}
```

#### 4. Build Docker Environment

```bash
docker-compose build
```

#### 5. Run Fine-tuning

```bash
# Update docker-compose.yml to mount your base model directory
# Then run:
docker-compose run --rm ai-trainer

# Inside container:
python finetune_omni_wrapper.py \
    --dataset /workspace/data/example_dataset.jsonl \
    --model-path /workspace/base_model \
    --output /workspace/models/qwen_omni_finetuned_lora \
    --max-steps 100 \
    --lr 2e-4
```

#### 6. Test the Model

```bash
# Inside container:
python test_finetuned_model.py
```

#### 7. Run Inference Server (Optional)

```bash
cd ai_server
pip install -r requirements.txt
python main.py
```

The server will start on `http://localhost:5000` with two endpoints:
- `POST /chat`: Free-form conversation
- `POST /analyze`: Structured analysis of data

### 🎯 Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-steps` | 100 | Number of training steps |
| `--lr` | 2e-4 | Learning rate |
| Batch size | 1 | Per-device batch size |
| Gradient accumulation | 4 | Effective batch size = 4 |
| LoRA rank | 16 | LoRA adapter rank |
| LoRA alpha | 16 | LoRA scaling factor |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | Layers to apply LoRA |
| Quantization | 4-bit (NF4) | Memory-efficient quantization |

### 📊 Results

- ✅ Training launches successfully
- ✅ Loss decreases (8.83 → lower)
- ✅ Only **0.87% of parameters trained** (41M/4.7B)
- ✅ Model **retains full multimodal architecture**
- ✅ QLoRA 4-bit: ~7.5GB VRAM used

### 🤝 Contributing

Contributions are welcome! Feel free to:
- Open issues for bugs or questions
- Submit pull requests with improvements
- Share your fine-tuning results

### 📝 License

This project is provided as-is for educational and research purposes.

### 🙏 Acknowledgments

- **Qwen Team** for the Qwen2.5-Omni model
- **Hugging Face** for Transformers, PEFT, and bitsandbytes

### 📧 Contact

For questions or discussions, feel free to open an issue on GitHub.

---

## Français

### Aperçu

**AITrainer** est un framework flexible de fine-tuning pour **modèles de langage multimodaux** qui **préserve toutes leurs capacités** (audio, vision, vidéo, texte). Conçu initialement pour Qwen2.5-Omni-3B, l'approche de wrapper personnalisé peut être adaptée à d'autres architectures multimodales qui ne fonctionnent pas avec les outils standards.

**Cas d'usage principal:** Qwen2.5-Omni-3B (testé et fonctionnel)
**Adaptable à:** Autres modèles multimodaux avec architectures complexes ou implémentations forward() non-standard

Le défi principal : Les outils de fine-tuning standard comme `SFTTrainer` ne fonctionnent pas avec certaines architectures multimodales. Ce projet fournit une solution fonctionnelle utilisant des wrappers personnalisés et l'application ciblée de LoRA.

### 🌟 Caractéristiques Principales

- ✅ **Préserve les capacités multimodales** (audio, vision, vidéo)
- ✅ **Fine-tuning QLoRA** avec quantification 4-bit (~7.5GB VRAM sur RTX 3080)
- ✅ **Wrapper personnalisé** exposant une méthode `forward()` pour l'entraînement
- ✅ **Seulement 0.87% des paramètres entraînés** (41M sur 4.7B)
- ✅ **Environnement Docker** avec CUDA 12.3.2 et PyTorch 2.6
- ✅ **Serveur FastAPI** pour l'inférence avec adaptateurs LoRA

### 📁 Structure du Projet

```
AITrainer/
├── Dockerfile                      # Environnement CUDA + PyTorch 2.6 + Transformers
├── docker-compose.yml              # Configuration Docker Compose
├── experiments/
│   └── qwen25_omni_custom/         # Approche wrapper personnalisé
│       ├── finetune_omni_wrapper.py  # Script d'entraînement principal
│       ├── inspect_model.py          # Inspecteur d'architecture du modèle
│       └── test_finetuned_model.py   # Script de test
├── ai_server/
│   ├── main.py                     # Serveur d'inférence FastAPI
│   └── requirements.txt            # Dépendances Python
├── data/
│   └── dataset_v3.jsonl            # Dataset d'entraînement (format JSONL)
└── models/                         # Répertoire pour les adaptateurs LoRA
```

### 🔧 Comment Ça Marche

#### Le Problème

Qwen2.5-Omni a une architecture inhabituelle :
- `Qwen2_5OmniForConditionalGeneration` contient :
  - **Thinker** : Le composant de modèle de langage
  - **Talker** : Composant de synthèse vocale
  - **Encodeurs multimodaux** : Processeurs vision, audio, vidéo
- Le `SFTTrainer` standard échoue car `forward()` n'est pas correctement implémenté

#### La Solution

1. **Wrapper Personnalisé** : Créer une classe `Qwen2_5OmniWrapper` qui :
   - Accède au composant interne `thinker`
   - Expose une méthode `forward()` appropriée pour l'entraînement texte uniquement
   - Délègue `generate()` au modèle Omni complet (préserve le multimodal)

2. **LoRA Avant Wrapping** (crucial) :
   ```python
   # 1. Extraire le Thinker
   thinker = omni_model.thinker

   # 2. Appliquer LoRA au Thinker
   thinker_with_lora = get_peft_model(thinker, lora_config)

   # 3. Remplacer le Thinker dans omni_model
   omni_model.thinker = thinker_with_lora

   # 4. ENSUITE créer le wrapper
   model = Qwen2_5OmniWrapper(omni_model)
   ```

3. **QLoRA** : Quantification 4-bit + adaptateurs LoRA pour tenir dans ~7.5GB VRAM

### 🚀 Démarrage Rapide

#### Prérequis

- Docker avec runtime NVIDIA
- GPU NVIDIA avec support CUDA (testé sur RTX 3080)
- Au moins 16GB RAM et 10GB VRAM

#### 1. Cloner et Configuration

```bash
git clone https://github.com/votreusername/AITrainer.git
cd AITrainer
```

#### 2. Télécharger le Modèle de Base

Téléchargez Qwen2.5-Omni-3B depuis Hugging Face :
```bash
# Créer un répertoire pour le modèle de base
mkdir -p base_model

# Télécharger avec git-lfs ou huggingface-cli
huggingface-cli download Qwen/Qwen2.5-Omni-3B-Instruct --local-dir base_model
```

#### 3. Préparer Votre Dataset

Créez un fichier JSONL dans `data/` avec vos exemples d'entraînement. Formats supportés :

**Chat simple :**
```json
{
  "type": "chat",
  "messages": [
    {"role": "user", "content": "Qu'est-ce que X ?"},
    {"role": "assistant", "content": "X est..."}
  ]
}
```

**Analyse avec contexte :**
```json
{
  "type": "analysis",
  "context": {"today_log": {"day_of_flower": 18}},
  "messages": [
    {"role": "user", "content": "Analyse ceci"},
    {"role": "assistant", "content": "Analyse..."}
  ]
}
```

**Analyse avec payload JSON :**
```json
{
  "type": "analysis_json",
  "context": {
    "session_config": {...},
    "products_catalog": {...},
    "today_log": {...}
  },
  "messages": [
    {"role": "user", "content": "Analyse ce payload"},
    {"role": "assistant", "content": "Analyse complète..."}
  ]
}
```

#### 4. Construire l'Environnement Docker

```bash
docker-compose build
```

#### 5. Lancer le Fine-tuning

```bash
# Mettre à jour docker-compose.yml pour monter votre répertoire de modèle de base
# Puis exécuter :
docker-compose run --rm ai-trainer

# Dans le conteneur :
python finetune_omni_wrapper.py \
    --dataset /workspace/data/dataset_v3.jsonl \
    --model-path /workspace/base_model \
    --output /workspace/models/qwen_omni_finetuned_lora \
    --max-steps 100 \
    --lr 2e-4
```

#### 6. Tester le Modèle

```bash
# Dans le conteneur :
python test_finetuned_model.py
```

#### 7. Lancer le Serveur d'Inférence (Optionnel)

```bash
cd ai_server
pip install -r requirements.txt
python main.py
```

Le serveur démarrera sur `http://localhost:5000` avec deux endpoints :
- `POST /chat` : Conversation libre
- `POST /analyze` : Analyse structurée de données

### 🎯 Configuration d'Entraînement

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `--max-steps` | 100 | Nombre d'étapes d'entraînement |
| `--lr` | 2e-4 | Taux d'apprentissage |
| Batch size | 1 | Taille de batch par device |
| Accumulation de gradient | 4 | Taille de batch effective = 4 |
| Rang LoRA | 16 | Rang de l'adaptateur LoRA |
| Alpha LoRA | 16 | Facteur d'échelle LoRA |
| Modules cibles | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | Couches où appliquer LoRA |
| Quantification | 4-bit (NF4) | Quantification économe en mémoire |

### 📊 Résultats

- ✅ L'entraînement démarre avec succès
- ✅ La loss diminue (8.83 → plus bas)
- ✅ Seulement **0.87% des paramètres entraînés** (41M/4.7B)
- ✅ Le modèle **conserve toute l'architecture multimodale**
- ✅ QLoRA 4-bit : ~7.5GB VRAM utilisé

### 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
- Ouvrir des issues pour des bugs ou questions
- Soumettre des pull requests avec des améliorations
- Partager vos résultats de fine-tuning

### 📝 Licence

Ce projet est fourni tel quel à des fins éducatives et de recherche.

### 🙏 Remerciements

- **Équipe Qwen** pour le modèle Qwen2.5-Omni
- **Hugging Face** pour Transformers, PEFT et bitsandbytes

### 📧 Contact

Pour des questions ou discussions, n'hésitez pas à ouvrir une issue sur GitHub.

---

## Italiano

### Panoramica

**AITrainer** è un framework flessibile di fine-tuning per **modelli linguistici multimodali** che **preserva tutte le loro capacità** (audio, visione, video, testo). Originariamente progettato per Qwen2.5-Omni-3B, l'approccio wrapper personalizzato può essere adattato ad altre architetture multimodali che non funzionano con strumenti standard.

**Caso d'uso principale:** Qwen2.5-Omni-3B (testato e funzionante)
**Adattabile a:** Altri modelli multimodali con architetture complesse o implementazioni forward() non-standard

La sfida principale: Gli strumenti di fine-tuning standard come `SFTTrainer` non funzionano con certe architetture multimodali. Questo progetto fornisce una soluzione funzionante utilizzando wrapper personalizzati e applicazione mirata di LoRA.

### 🌟 Caratteristiche Principali

- ✅ **Preserva le capacità multimodali** (audio, visione, video)
- ✅ **Fine-tuning QLoRA** con quantizzazione 4-bit (~7.5GB VRAM su RTX 3080)
- ✅ **Wrapper personalizzato** che espone il metodo `forward()` per il training
- ✅ **Solo lo 0.87% dei parametri addestrati** (41M su 4.7B)
- ✅ **Ambiente Docker** con CUDA 12.3.2 e PyTorch 2.6
- ✅ **Server FastAPI** per inferenza con adattatori LoRA

### 📁 Struttura del Progetto

```
AITrainer/
├── Dockerfile                      # Ambiente CUDA + PyTorch 2.6 + Transformers
├── docker-compose.yml              # Configurazione Docker Compose
├── experiments/
│   └── qwen25_omni_custom/         # Approccio wrapper personalizzato
│       ├── finetune_omni_wrapper.py  # Script di training principale
│       ├── inspect_model.py          # Ispettore architettura modello
│       └── test_finetuned_model.py   # Script di test
├── ai_server/
│   ├── main.py                     # Server di inferenza FastAPI
│   └── requirements.txt            # Dipendenze Python
├── data/
│   └── dataset_v3.jsonl            # Dataset di training (formato JSONL)
└── models/                         # Directory per adattatori LoRA
```

### 🔧 Come Funziona

#### Il Problema

Qwen2.5-Omni ha un'architettura insolita:
- `Qwen2_5OmniForConditionalGeneration` contiene:
  - **Thinker**: Il componente del modello linguistico
  - **Talker**: Componente di sintesi vocale
  - **Encoder multimodali**: Processori visione, audio, video
- Lo `SFTTrainer` standard fallisce perché `forward()` non è correttamente implementato

#### La Soluzione

1. **Wrapper Personalizzato**: Creare una classe `Qwen2_5OmniWrapper` che:
   - Accede al componente interno `thinker`
   - Espone un metodo `forward()` appropriato per il training solo testo
   - Delega `generate()` al modello Omni completo (preserva multimodale)

2. **LoRA Prima del Wrapping** (cruciale):
   ```python
   # 1. Estrarre il Thinker
   thinker = omni_model.thinker

   # 2. Applicare LoRA al Thinker
   thinker_with_lora = get_peft_model(thinker, lora_config)

   # 3. Sostituire il Thinker in omni_model
   omni_model.thinker = thinker_with_lora

   # 4. POI creare il wrapper
   model = Qwen2_5OmniWrapper(omni_model)
   ```

3. **QLoRA**: Quantizzazione 4-bit + adattatori LoRA per rientrare in ~7.5GB VRAM

### 🚀 Avvio Rapido

#### Prerequisiti

- Docker con runtime NVIDIA
- GPU NVIDIA con supporto CUDA (testato su RTX 3080)
- Almeno 16GB RAM e 10GB VRAM

#### 1. Clonare e Configurazione

```bash
git clone https://github.com/tuousername/AITrainer.git
cd AITrainer
```

#### 2. Scaricare il Modello Base

Scarica Qwen2.5-Omni-3B da Hugging Face:
```bash
# Creare una directory per il modello base
mkdir -p base_model

# Scaricare con git-lfs o huggingface-cli
huggingface-cli download Qwen/Qwen2.5-Omni-3B-Instruct --local-dir base_model
```

#### 3. Preparare il Dataset

Crea un file JSONL in `data/` con i tuoi esempi di training. Formati supportati:

**Chat semplice:**
```json
{
  "type": "chat",
  "messages": [
    {"role": "user", "content": "Cos'è X?"},
    {"role": "assistant", "content": "X è..."}
  ]
}
```

**Analisi con contesto:**
```json
{
  "type": "analysis",
  "context": {"today_log": {"day_of_flower": 18}},
  "messages": [
    {"role": "user", "content": "Analizza questo"},
    {"role": "assistant", "content": "Analisi..."}
  ]
}
```

**Analisi con payload JSON:**
```json
{
  "type": "analysis_json",
  "context": {
    "session_config": {...},
    "products_catalog": {...},
    "today_log": {...}
  },
  "messages": [
    {"role": "user", "content": "Analizza questo payload"},
    {"role": "assistant", "content": "Analisi completa..."}
  ]
}
```

#### 4. Costruire l'Ambiente Docker

```bash
docker-compose build
```

#### 5. Avviare il Fine-tuning

```bash
# Aggiornare docker-compose.yml per montare la directory del modello base
# Poi eseguire:
docker-compose run --rm ai-trainer

# All'interno del container:
python finetune_omni_wrapper.py \
    --dataset /workspace/data/dataset_v3.jsonl \
    --model-path /workspace/base_model \
    --output /workspace/models/qwen_omni_finetuned_lora \
    --max-steps 100 \
    --lr 2e-4
```

#### 6. Testare il Modello

```bash
# All'interno del container:
python test_finetuned_model.py
```

#### 7. Avviare il Server di Inferenza (Opzionale)

```bash
cd ai_server
pip install -r requirements.txt
python main.py
```

Il server si avvierà su `http://localhost:5000` con due endpoint:
- `POST /chat`: Conversazione libera
- `POST /analyze`: Analisi strutturata di dati

### 🎯 Configurazione Training

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `--max-steps` | 100 | Numero di passi di training |
| `--lr` | 2e-4 | Learning rate |
| Batch size | 1 | Dimensione batch per dispositivo |
| Accumulo gradiente | 4 | Dimensione batch effettiva = 4 |
| Rango LoRA | 16 | Rango adattatore LoRA |
| Alpha LoRA | 16 | Fattore di scala LoRA |
| Moduli target | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | Layer dove applicare LoRA |
| Quantizzazione | 4-bit (NF4) | Quantizzazione efficiente in memoria |

### 📊 Risultati

- ✅ Il training si avvia con successo
- ✅ La loss diminuisce (8.83 → inferiore)
- ✅ Solo **0.87% dei parametri addestrati** (41M/4.7B)
- ✅ Il modello **mantiene tutta l'architettura multimodale**
- ✅ QLoRA 4-bit: ~7.5GB VRAM utilizzati

### 🤝 Contribuire

I contributi sono benvenuti! Sentiti libero di:
- Aprire issue per bug o domande
- Inviare pull request con miglioramenti
- Condividere i tuoi risultati di fine-tuning

### 📝 Licenza

Questo progetto è fornito così com'è per scopi educativi e di ricerca.

### 🙏 Ringraziamenti

- **Team Qwen** per il modello Qwen2.5-Omni
- **Hugging Face** per Transformers, PEFT e bitsandbytes

### 📧 Contatto

Per domande o discussioni, sentiti libero di aprire un'issue su GitHub.

---

**Made with ❤️ by the AI community**
