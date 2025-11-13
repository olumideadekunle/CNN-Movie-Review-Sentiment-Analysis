# CNN-Movie-Review-Sentiment-Analysis
A Convolutional Neural Network (CNN) implementation for sentiment analysis on movie reviews, showcasing text preprocessing, model training, and evaluation with visualizations


# Movie Review Sentiment Analysis with CNN

This project implements a Convolutional Neural Network (CNN) for sentiment analysis on movie reviews. It demonstrates the complete workflow from data loading and preprocessing to model training, evaluation, and making predictions on new text.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Evaluation Metrics](#evaluation-metrics)
- [Insights and Future Work](#insights-and-future-work)
- [License](#license)

## Project Overview
This notebook showcases a basic sentiment analysis system built with a 1D CNN using TensorFlow/Keras. It's designed to classify movie reviews as either 'positive' or 'negative'. For demonstration purposes, a small synthetic dataset is used, allowing for a quick run-through of the entire machine learning pipeline.

## Features
- **Data Loading & Preprocessing**: Loading of text data and cleaning steps including HTML tag stripping, accent removal, contractions expansion, lowercasing, and removal of special characters.
- **Tokenization & Padding**: Conversion of text reviews into numerical sequences using Keras's Tokenizer, followed by sequence padding to ensure uniform input length for the CNN model.
- **CNN Model Implementation**: A Sequential Keras model featuring an Embedding layer, multiple Conv1D and MaxPooling1D layers, a Flatten layer, and Dense layers for classification.
- **Model Training & Evaluation**: Training the CNN model and evaluating its performance using accuracy and loss metrics.
- **Text Length Distribution Visualization**: Histograms of review lengths for both training and testing datasets, along with statistical summaries, to understand data distribution and justify `MAX_SEQUENCE_LENGTH`.
- **Word Cloud Generation**: Visual representation of the most frequent words in the training data, offering quick insights into common vocabulary.
- **Model Architecture Diagram**: A visual diagram of the CNN model's layers and connections, aiding in understanding the network structure.
- **Confusion Matrix Heatmap**: An intuitive heatmap visualization of the confusion matrix to assess model performance in distinguishing between sentiment classes.
- **Robust Prediction Pipeline**: A utility function to seamlessly preprocess raw text and predict sentiment using the trained model.

## Setup and Installation
To run this notebook, you'll need a Python environment, preferably within Google Colab for easy setup.

1.  **Open in Google Colab**: Click the "Open in Colab" badge (if available) or upload the `.ipynb` file to Google Colab.
2.  **Install Dependencies**: The notebook automatically installs necessary libraries. Run the first few cells to ensure `contractions`, `textsearch`, `tqdm`, `nltk`, `wordcloud`, `pydot`, `graphviz`, `seaborn` and `scikit-learn` are installed.
    ```bash
    !pip install contractions textsearch tqdm
    ! pip install wordcloud pydot graphviz seaborn scikit-learn
    import nltk
    nltk.download('punkt')
    ```
3.  **Synthetic Dataset**: The project uses an in-memory synthetic dataset. No external file download is required to get started.

## Usage
1.  **Run All Cells**: Execute all cells in the notebook sequentially from top to bottom.
2.  **Explore**: Review the output of each cell to understand the data preprocessing steps, model training progress, and evaluation results.
3.  **Make Predictions**: Use the `predict_sentiment_pipeline` function to test the model with your own review texts.
    ```python
    # Example usage
    from tensorflow.keras.preprocessing.sequence import pad_sequences # Ensure this is imported

    new_review = "This movie was phenomenal!"
    predicted_label, probability = predict_sentiment_pipeline(new_review)
    print(f"Review: '{new_review}'\nPredicted Sentiment: {predicted_label} (Probability: {probability:.4f})")
    ```

## Model Architecture
The CNN model consists of:
-   **Embedding Layer**: Maps words to dense vector representations.
-   **Conv1D Layers**: Extract local features from the sequences.
-   **MaxPooling1D Layers**: Downsample the feature maps.
-   **Flatten Layer**: Converts the 3D output of convolutional layers into a 1D vector.
-   **Dense Layers**: Fully connected layers for classification.
-   **Sigmoid Activation**: Output layer for binary classification.

An architectural diagram is generated within the notebook for a visual representation.

## Evaluation Metrics
-   **Accuracy**: Overall correctness of the model's predictions.
-   **Loss**: Binary Crossentropy, measuring the error between predicted probabilities and actual labels.
-   **Confusion Matrix**: Visualized as a heatmap, showing true positives, true negatives, false positives, and false negatives.

## Insights and Future Work

Due to the use of a small synthetic dataset for demonstration:
-   **`MAX_SEQUENCE_LENGTH`**: The current `MAX_SEQUENCE_LENGTH` of `1000` is highly inefficient for this synthetic data, leading to excessive padding. For real-world applications, it should be optimized based on the actual distribution of review lengths (e.g., using the 90th or 95th percentile) to balance information retention and computational efficiency.
-   **Model Performance**: The model's performance on the synthetic data, especially with the prediction pipeline, indicates potential limitations and biases (e.g., misclassifying a clearly negative review as positive). This is expected with minimal training data.

**Next Steps for Improvement:**
1.  **Real-world Dataset**: Train and evaluate the model on a larger, more diverse, and representative real-world movie review dataset (e.g., IMDB movie review dataset) to improve generalization and accuracy.
2.  **Hyperparameter Tuning**: Experiment with different CNN architectures, filter sizes, number of filters, pooling strategies, and dense layer sizes.
3.  **Advanced Preprocessing**: Incorporate stop word removal, stemming, or lemmatization.
4.  **Pre-trained Embeddings**: Utilize pre-trained word embeddings like Word2Vec or GloVe for better semantic understanding.

## License
This project is open-sourced under the MIT License. See the `LICENSE` file (if applicable) for more details.
