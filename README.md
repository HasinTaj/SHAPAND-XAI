Explainable Arabic Medical Text Classification
This project implements a robust framework for interpreting Transformer-based models applied to Arabic medical news classification. It focuses on bridging the gap between high-performance deep learning and human-understandable clinical decision-making by utilizing state-of-the-art eXplainable AI (XAI) techniques.
📌 Project Overview
The core objective is to classify and explain medical news articles (sourced from the Dubai Health Authority and Ministry of Health) using Fine-tuned Transformers. The project addresses the "black-box" nature of NLP models by identifying specific linguistic drivers that influence classification outcomes in the Arabic language.
🛠️ Key Features
Dual-Interpretability: Combines SHAP (Shapley Additive Explanations) for feature attribution and Attention Map Analysis to visualize internal model weights.
Long-Text Handling: Implements a custom dynamic truncation wrapper to process articles exceeding the standard 512-token Transformer limit (handling sequences up to 1,247 tokens).
Faithfulness Testing: Includes an Ablation-based evaluation framework that measures the drop in model confidence when top-weighted features are masked, ensuring explanation accuracy.
Interactive UI: Features a Gradio-powered dashboard for real-time model testing and visualization of influential medical keywords.
📊 Dataset
The dataset consists of Modern Standard Arabic (MSA) news articles focused on vaccinations, infectious diseases, and public health initiatives. Key entities include:

Clinical terms: (Vaccinations/التطعيمات, Immunity/المناعة).

Organizations: (Dubai Health Authority/هيئة الصحة بدبي).
🚀 Technical Stack
Language: Python

Models: Hugging Face Transformers (DistilBERT/AraBERT)

XAI Library: SHAP (Partition & Text Explainers)
🧪 Results & Analysis
The model demonstrated high sensitivity to clinical terminology. Through SHAP analysis, the project successfully identified that the model prioritizes semantic medical roots over morphological noise. Quantitative Faithfulness Scores validated that masking top-weighted clinical keywords significantly shifted the model's classification confidence, proving the transparency of the decision-making process.

Visualization: BertViz (Attention Maps), Gradio (UI)

Data Analysis: Pandas, NumPy, PyTorch
