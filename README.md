# 🎭 Sentiment Slang Analyzer

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135.1-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**A powerful FastAPI-based web application for sentiment analysis on social media text containing slang and informal language.**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [API Documentation](#-api-documentation) • [Model Architecture](#-model-architecture)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [API Documentation](#-api-documentation)
- [Model Architecture](#-model-architecture)
- [Dataset](#-dataset)
- [Docker Deployment](#-docker-deployment)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌟 Overview

**Sentiment Slang Analyzer** is an advanced machine learning application designed to analyze sentiment in social media text that contains slang, abbreviations, and informal language. The system leverages deep learning models (Bi-LSTM) to accurately classify text sentiment while handling the challenges of modern internet communication.

### Key Highlights

- 🔄 **Multi-Model Architecture**: Compare performance between baseline and enhanced models
- 🧹 **Advanced Text Preprocessing**: Handles emojis, slang, URLs, and special characters
- 🚀 **Fast & Scalable**: Built with FastAPI for high-performance API responses
- 🎨 **Interactive UI**: Clean, responsive web interface for real-time predictions
- 🐳 **Docker Ready**: Containerized deployment for easy scaling
- 📊 **Model Versioning**: Switch between different model versions seamlessly

---

## ✨ Features

### Core Functionality

- **Real-time Sentiment Analysis**: Instant prediction of positive, negative, or neutral sentiment
- **Slang Recognition**: Specialized preprocessing for social media language patterns
- **Confidence Scores**: Provides probability scores for prediction confidence
- **Model Comparison**: Test and compare different model versions side-by-side
- **RESTful API**: Easy integration with other applications
- **Responsive Design**: Works seamlessly on desktop and mobile devices

### Technical Features

- **Bi-LSTM Neural Networks**: Captures context from both directions in text
- **Custom Tokenization**: Vocabulary built from social media datasets
- **Emoji Processing**: Converts emojis to meaningful text representations
- **Text Normalization**: Handles repeated characters, special symbols, and URLs
- **Model Metadata**: Each model version includes performance metrics and descriptions

---

## 🛠️ Tech Stack

### Backend
- **FastAPI** - Modern, high-performance web framework
- **TensorFlow/Keras** - Deep learning model development
- **Python 3.11** - Core programming language
- **Uvicorn** - ASGI server for production

### Machine Learning
- **Bi-LSTM** - Bidirectional Long Short-Term Memory networks
- **Word Embeddings** - Dense vector representations
- **Keras Preprocessing** - Text tokenization and sequence processing

### Frontend
- **HTML5/CSS3** - Modern web standards
- **Vanilla JavaScript** - No framework dependencies
- **Responsive Design** - Mobile-first approach

### DevOps
- **Docker** - Containerization
- **Docker Compose** - Multi-container orchestration

---

## 📦 Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager
- (Optional) Docker and Docker Compose

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/debanjan-mondal-2005/Sentiment_Analyzer.git
   cd Sentiment_Analyzer
   ```

2. **Create a virtual environment**
   ```bash
   # Windows
   python -m venv sentiment_slang
   sentiment_slang\Scripts\activate

   # Linux/Mac
   python3 -m venv sentiment_slang
   source sentiment_slang/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify model files**
   ```bash
   # Ensure models directory contains:
   # - models/version1/sentiment_model/
   # - models/version2/sentiment_model/
   # Or run the setup script to create model versions
   python setup_model_versions.py
   ```

---

## 🚀 Usage

### Running the Application

1. **Start the FastAPI server**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Access the web interface**
   - Open your browser and navigate to: `http://localhost:8000`

3. **Using the API**
   - API documentation: `http://localhost:8000/docs`
   - Interactive API testing: `http://localhost:8000/redoc`

### Training New Models

**Train Version 1 (CPU-optimized)**
```bash
python train_cpu_model.py
```

**Train Version 2 (Enhanced model)**
```bash
python train_model_v2.py
```

---

## 📁 Project Structure

```
Sentiment_Analyzer/
├── app/
│   ├── __init__.py              # Package initialization
│   ├── main.py                  # FastAPI application & routes
│   ├── preprocessing.py         # Text preprocessing utilities
│   ├── utils.py                 # Single model utilities
│   ├── utils_multimodel.py      # Multi-model management
│   └── static/
│       ├── index.html           # Web interface
│       ├── script.js            # Frontend logic
│       └── style.css            # Styling
├── data/
│   ├── social_media_sentiment_train.csv    # Training dataset
│   └── social_media_sentiment_test.csv     # Test dataset
├── models/
│   ├── version1/                # Baseline model
│   │   ├── metadata.json
│   │   └── sentiment_model/
│   └── version2/                # Enhanced model
│       ├── metadata.json
│       └── sentiment_model/
├── docker-compose.yml           # Docker orchestration
├── dockerfile                   # Container definition
├── requirements.txt             # Python dependencies
├── train_cpu_model.py          # Model training script (v1)
├── train_model_v2.py           # Model training script (v2)
├── setup_model_versions.py     # Model setup utility
├── PROJECT_FLOWCHART.md        # System architecture diagram
└── README.md                   # Project documentation
```

---

## 📡 API Documentation

### Endpoints

#### **GET /**
Serves the main web interface

**Response:** HTML page

---

#### **GET /models**
Retrieves available model versions and metadata

**Response:**
```json
{
  "version1": {
    "available": true,
    "metadata": {
      "name": "Baseline Sentiment Model",
      "description": "Standard Bi-LSTM model for sentiment analysis",
      "training_date": "2026-03-01",
      "accuracy": 0.85
    }
  },
  "version2": {
    "available": true,
    "metadata": {
      "name": "Enhanced Sentiment Model",
      "description": "Improved model with better slang handling",
      "training_date": "2026-03-05",
      "accuracy": 0.89
    }
  }
}
```

---

#### **POST /predict**
Performs sentiment analysis on input text

**Request Body:**
```json
{
  "text": "This is awesome! 😊 Can't wait to see more",
  "model_version": "version2"
}
```

**Response:**
```json
{
  "sentiment": "Positive",
  "confidence": 0.94,
  "model_version": "version2",
  "model_info": {
    "name": "Enhanced Sentiment Model",
    "accuracy": 0.89
  },
  "cleaned_text": "this is awesome cant wait to see more"
}
```

**Status Codes:**
- `200 OK` - Successful prediction
- `400 Bad Request` - Invalid input
- `404 Not Found` - Model version not found
- `500 Internal Server Error` - Server error

---

## 🧠 Model Architecture

### Bi-LSTM Neural Network

Both model versions use a Bidirectional LSTM architecture optimized for sequence processing:

```
Input Layer (Text Sequences)
        ↓
Embedding Layer (128 dimensions)
        ↓
Bidirectional LSTM (64 units)
        ↓
Dropout Layer (0.5)
        ↓
Dense Layer (Softmax activation)
        ↓
Output (Sentiment Classes)
```

### Model Specifications

| Component | Version 1 | Version 2 |
|-----------|-----------|-----------|
| **Vocabulary Size** | 5,000 | 10,000 |
| **Max Sequence Length** | 30 | 30 |
| **Embedding Dimension** | 128 | 128 |
| **LSTM Units** | 64 | 64 |
| **Dropout Rate** | 0.5 | 0.5 |
| **Training Epochs** | 10 | 15 |
| **Batch Size** | 32 | 32 |
| **Optimizer** | Adam | Adam |

### Preprocessing Pipeline

1. **Text Cleaning**
   - Convert to lowercase
   - Remove URLs and mentions
   - Handle special characters
   - Normalize repeated characters

2. **Emoji Processing**
   - Convert emojis to text descriptions
   - Preserve sentiment information

3. **Slang Normalization**
   - Expand common abbreviations
   - Handle internet slang patterns

4. **Tokenization**
   - Convert text to sequences
   - Pad sequences to fixed length

---

## 📊 Dataset

The model is trained on social media sentiment datasets containing:

- **Training samples**: ~10,000+ texts
- **Test samples**: ~2,000+ texts
- **Sentiment classes**: Positive, Negative, Neutral
- **Source**: Social media platforms (Twitter, Reddit, etc.)
- **Characteristics**: Contains slang, emojis, abbreviations, and informal language

### Data Distribution

The dataset is balanced across sentiment classes to prevent bias:
- Positive: ~33%
- Negative: ~33%
- Neutral: ~34%

---

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

1. **Build and run**
   ```bash
   docker-compose up --build
   ```

2. **Access the application**
   - Web Interface: `http://localhost:8000`
   - API Docs: `http://localhost:8000/docs`

3. **Stop the application**
   ```bash
   docker-compose down
   ```

### Using Docker Directly

1. **Build the image**
   ```bash
   docker build -t sentiment-analyzer .
   ```

2. **Run the container**
   ```bash
   docker run -p 8000:8000 sentiment-analyzer
   ```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add amazing feature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines for Python code
- Add unit tests for new features
- Update documentation for API changes
- Ensure Docker build succeeds before submitting PR

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Debanjan Mondal**

- GitHub: [@debanjan-mondal-2005](https://github.com/debanjan-mondal-2005)
- Project Link: [Sentiment_Analyzer](https://github.com/debanjan-mondal-2005/Sentiment_Analyzer)

---

## 🙏 Acknowledgments

- TensorFlow and Keras teams for excellent deep learning frameworks
- FastAPI community for the modern web framework
- Social media datasets contributors
- Open source community for inspiration and support

---

## 📧 Contact & Support

For questions, issues, or suggestions:

- 🐛 **Report bugs**: [GitHub Issues](https://github.com/debanjan-mondal-2005/Sentiment_Analyzer/issues)
- 💡 **Feature requests**: [GitHub Discussions](https://github.com/debanjan-mondal-2005/Sentiment_Analyzer/discussions)
- 📧 **Email**: Contact through GitHub profile

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ by Debanjan Mondal

</div>
