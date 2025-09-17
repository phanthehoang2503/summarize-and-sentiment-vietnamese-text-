# Vietnamese Text Summarization and Sentiment Analysis

In this project, I developed a text summarization and sentiment analysis system based on a Vietnamese dataset. The project provided me with practical experience in data preprocessing, selecting and tuning trainer parameters to optimize model performance, constructing pipelines for seamless interaction between modules, and deploying the models within a web-based application for user interaction.

## Key Features

- **Text Summarization**: Smart Vietnamese text summarization with 20-30% compression ratio
- **Sentiment Analysis**: Accurate sentiment classification for Vietnamese text
- **Smart Pipeline**: Intelligent text processing with dynamic parameter adjustment
- **YAML Configuration**: Flexible configuration management system
- **Data Processing**: Robust preprocessing pipeline for Vietnamese text


### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Installation

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure the project**
   ```python
   from config import config
   # Configuration is automatically loaded from params.yaml
   ```

#### Text Summarization
```python
from src.models.summarizer import create_summarizer

summarizer = create_summarizer()
text = "Your Vietnamese text here..."
summary = summarizer.summarize(text)
print(summary)
```

#### Sentiment Analysis
```python
from src.models.sentiment import create_sentiment_analyzer

analyzer = create_sentiment_analyzer()
comment = "Your Vietnamese comment here..."
result = analyzer.predict_sentiment(comment, return_probabilities=True)
print(result['predicted_label'], result['confidence'])
```

#### Combined Pipeline
```python
from src.models.pipeline import create_pipeline

pipeline = create_pipeline()
text = "Your Vietnamese text here..."
result = pipeline.analyze(text, include_original_text=False)
print(result['summary'])
print(result['sentiment']['predicted_label'])
```

## Model Performance

### Summarization Model
- **Architecture**: Vietnamese T5 (VietAI/vit5-base)
- **Compression Ratio**: 20-30% of original text
- **Languages**: Vietnamese
- **Quality**: High coherence and fluency

### Sentiment Analysis Model
- **Architecture**: PhoBERT (vinai/phobert-base)
- **Accuracy**: 95%+ on Vietnamese sentiment data
- **Classes**: Positive, Negative, Neutral


## Examples

Check out the demo scripts for practical examples:
- `scripts/demos/demo_summarizer.py` - Text summarization examples
- `scripts/demos/demo_sentiment.py` - Sentiment analysis examples
- `scripts/demos/demo_pipeline.py` - Combined pipeline examples

## License
This project is licensed under the MIT License.