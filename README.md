# 🎵 Music Genre Classification Model

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

> A machine learning system that automatically classifies songs into 24 distinct musical genres using audio features, metadata, and lyrical content analysis.

## 📈 Performance Highlights

| Metric | Value | Notes |
|--------|--------|-------|
| **Accuracy** | 27.95% | 6.7x better than random (4.17%) |
| **Genres Classified** | 24 | From hip-hop to classical |
| **Dataset Size** | 551,443 songs | Balanced to 36,005 samples |
| **Overfitting Control** | 3.6% gap | Production-ready stability |

## 🎯 Best Performing Genres

```
🏆 Trap Music      87.4% accuracy
🥈 Heavy Metal     60.7% accuracy  
🥉 Pop Music       54.7% accuracy
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/music-genre-classifier.git
cd music-genre-classifier

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
import joblib
import pandas as pd

# Load trained model
model = joblib.load('models/music_genre_classifier.pkl')
scaler = joblib.load('models/feature_scaler.pkl')

# Prepare song features
song_data = {
    'Tempo': 120,
    'Energy': 75,
    'Danceability': 60,
    'Acousticness': 20,
    'Loudness': -8.5,
    'Speechiness': 15,
    'has_love': 1,
    'has_party': 0,
    # ... other features
}

# Make prediction
features = scaler.transform([list(song_data.values())])
predicted_genre = model.predict(features)[0]
confidence = model.predict_proba(features)[0].max()

print(f"Predicted Genre: {predicted_genre}")
print(f"Confidence: {confidence:.2%}")
```

## 📁 Project Structure

```
music-genre-classifier/
├── 📊 data/
│   ├── raw/                    # Original dataset
│   ├── processed/              # Cleaned and balanced data
│   └── sample/                 # Sample data for testing
├── 📓 notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
├── 🤖 models/
│   ├── music_genre_classifier.pkl
│   ├── feature_scaler.pkl
│   └── selected_features.txt
├── 📜 src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── utils.py
├── 📋 requirements.txt
├── 📖 README.md
└── 📄 LICENSE
```

## 🔍 Model Architecture

### Genre Categories (24 total)
```
🎤 Hip-hop Family    🎸 Rock Family        🎹 Electronic
├── hip_hop          ├── rock              ├── electronic
├── rap              ├── alternative_rock  ├── house
└── trap             ├── indie_rock        ├── techno
                     ├── punk_rock         └── dubstep
🎵 Pop & Mainstream  ├── heavy_metal       
├── pop              └── metal             🎼 Traditional
├── indie_pop                              ├── country
└── k_pop            🎺 Jazz & Blues       ├── folk
                     ├── jazz              ├── classical
                     ├── blues             ├── reggae
                     └── soul              └── other
```

### Feature Engineering

| Feature Type | Count | Examples |
|--------------|-------|----------|
| **Audio Features** | 12 | Tempo, Energy, Danceability, Acousticness |
| **Lyrical Features** | 16 | has_love, has_party, has_money, has_heart |
| **Metadata** | 10 | Explicit content, playlist tags, popularity |

### Top Predictive Features

1. 🚫 **Explicit Content** (15.76%) - Strong genre indicator
2. 🎸 **Acousticness** (13.27%) - Acoustic vs electronic separation  
3. 🗣️ **Speechiness** (11.05%) - Rap/hip-hop identification
4. ⚡ **Energy** (10.41%) - High-energy vs calm genres
5. 💃 **Danceability** (9.75%) - Party vs contemplative music

## 📊 Performance Analysis

### Confusion Matrix (Top Genres)
```
                Predicted
Actual      pop  alt_rock  heavy_metal  rock  hip_hop  trap
pop         328      34          15    88       47    43
alt_rock    125     140         111    81       23    23  
heavy_metal  38      50         364    43       22    22
rock        184      53          55   202       32    33
hip_hop     130      35          25    43      112   209
trap         24       7           3     3       29   486
```

### Genre Performance Tiers

**🟢 Excellent (>50%)**
- Trap, Heavy Metal, Pop

**🟡 Good (25-50%)**  
- Rock, Metal, Alternative Rock

**🔴 Challenging (<25%)**
- Hip-hop variants, Indie genres

## 🛠️ Development Workflow

### 1. Data Preprocessing
```python
# Genre consolidation (3097 → 31 → 24 genres)
python src/data_preprocessing.py --input data/raw/ --output data/processed/

# Class balancing and sampling
python src/feature_engineering.py --balance-classes --sample-size 100000
```

### 2. Model Training
```python
# Train ensemble models
python src/model_training.py --models rf,gb,lr --cv-folds 5

# Hyperparameter optimization  
python src/model_training.py --optimize --search-space config/param_grid.json
```

### 3. Evaluation
```python
# Generate performance reports
python src/evaluation.py --model models/best_model.pkl --test-data data/test.csv
```

## 📈 Results Deep Dive

### Business Impact
- **✅ High-confidence applications:** Trap identification (87%), Heavy metal detection (61%)
- **⚠️ Moderate confidence:** Pop categorization (55%), General rock classification (34%)
- **❌ Challenging areas:** Hip-hop ecosystem disambiguation, Rock subgenre distinction

### Technical Achievements
- **Memory Efficiency:** Reduced 1.9GB dataset to manageable size
- **Overfitting Control:** Advanced regularization techniques
- **Feature Selection:** Streamlined from 38 to 28 optimal features
- **Cross-Validation:** Robust 5-fold validation with stratification

## 🚧 Known Limitations

| Issue | Impact | Mitigation |
|-------|--------|------------|
| **Genre Subjectivity** | High | Focus on clear-cut distinctions |
| **Hip-hop Confusion** | Medium | Implement hierarchical classification |
| **Small Genre Bias** | Medium | Balanced sampling strategies |
| **Lyrical Underutilization** | Low | Advanced NLP in future versions |

## 🔮 Future Improvements

### Phase 1: Quick Wins (Target: 35-45%)
- [ ] **Ensemble Methods** - Combine RF + XGBoost + Neural Net
- [ ] **Full Dataset Usage** - Cloud computing for 551K samples  
- [ ] **Feature Interactions** - Energy×Loudness combinations

### Phase 2: Major Enhancements (Target: 45-60%)
- [ ] **Hierarchical Classification** - Two-stage genre→subgenre
- [ ] **Advanced NLP** - Word embeddings and sentiment analysis
- [ ] **Deep Learning** - Neural networks for complex patterns

### Phase 3: State-of-Art (Target: 60-75%)
- [ ] **Raw Audio Processing** - Spectrograms and MFCCs
- [ ] **Multi-Modal Learning** - Audio + lyrics + metadata fusion
- [ ] **Real-time API** - Production deployment infrastructure

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black src/ tests/
flake8 src/ tests/
```

## 📚 Documentation

- [📊 **Data Dictionary**](docs/data_dictionary.md) - Feature descriptions and ranges
- [🔬 **Model Documentation**](docs/model_details.md) - Architecture and training details  
- [📈 **Performance Report**](docs/performance_analysis.md) - Detailed results analysis
- [🚀 **Deployment Guide**](docs/deployment.md) - Production setup instructions
- [🎯 **Use Cases**](docs/use_cases.md) - Real-world applications

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.