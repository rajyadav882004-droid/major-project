# major-project
Depression Detection Using Machine Learning

Depression Detection Using Machine Learning

Depression Detection Using Machine Learning is an educational AI/ML project designed to demonstrate how machine learning, Natural Language Processing (NLP), and an interactive web application can be combined for mental-wellness assessment. The project provides two complementary modules: a 20-question interactive assessment and an NLP-based text classification system using DistilBERT.

The web application is developed using Python and Streamlit, providing a clean and interactive interface where users can answer 20 questions related to mood, energy, concentration, sleep, loneliness, social interaction, daily activities, and emotional well-being. Each response contributes to an overall score, which is categorized into lower, moderate, or higher reported symptom burden. The scoring system is intended for educational demonstration and is not a clinically validated diagnostic instrument.

The second component uses a fine-tuned DistilBERT transformer model for text classification. Users can enter natural-language text describing their feelings, and the model processes the input using NLP techniques before predicting one of three dataset categories: Not Depressed, Moderately Depressed, or Severely Depressed. The model was trained using PyTorch, Hugging Face Transformers, and a depression-related text classification dataset.

The complete machine-learning workflow includes data preprocessing, tokenization, model training, validation, testing, evaluation, prediction, and web integration. Model performance is examined using metrics such as accuracy, precision, recall, F1-score, classification reports, and a confusion matrix.

The project is designed to run on a CPU-based environment, making it suitable for academic demonstration and experimentation without requiring high-end GPU hardware.

🛠️ Technologies Used

Python • Streamlit • PyTorch • Hugging Face Transformers • DistilBERT • Pandas • NumPy • Scikit-learn • NLP

⚠️ Disclaimer: This project is an academic prototype for educational purposes. Its results should not be considered a medical diagnosis or a replacement for qualified mental-health professionals.
