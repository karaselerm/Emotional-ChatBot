# Emotion-aware Conversational Chatbot

## Methods

### 1. **Emotion Classification**

The chatbot detects the emotional state of the user in real time using a dedicated emotion classification module.

* **Model:** Fine-tuned BERT (Bidirectional Encoder Representations from Transformers) on a labeled Russian dialogue/emotion dataset.
* **Workflow:** Each user message is tokenized and fed into the BERT model, which outputs one of several emotion labels (e.g., *positive*, *sad*, *anger*, *neutral*).
* **Improvements:** Various data balancing techniques were explored (oversampling, undersampling, synthetic data augmentation) to handle class imbalance and boost classification performance.

### 2. **Dialogue Generation (LLM-powered)**

To provide rich, supportive, and human-like replies, the bot uses an advanced Large Language Model:

* **API Integration:** OpenAI GPT-4o is called via API with a specially crafted prompt containing detected emotion, dialogue history, and stylistic instructions.
* **Prompt Engineering:** The system prompt guides the LLM to avoid generic answers, use empathy, and produce short, lively responses tailored to the user’s current emotional state.
* **Adaptability:** The approach allows to quickly swap out the backend LLM or fine-tune prompts for other languages or tone-of-voice.

### 3. **Dialogue Scenarios & User Experience**

* **Session Logic:** Each chat with the bot is tracked as a session, preserving history and enabling more coherent, context-aware responses.
* **Rating Mechanism:** After each session, users can rate the quality of the conversation, providing real feedback for iterative improvement and evaluation.
* **Scenario Simulation:** Synthetic dialogues and stress tests were used to benchmark bot responses under various emotional and conversational conditions.

### 4. **Data Engineering and Experimentation**

* **Jupyter Notebooks:** All data analysis, model training, and experiment tracking are done in organized notebooks, covering:

  * Exploratory Data Analysis (EDA)
  * Data preprocessing (tokenization, cleaning, class balancing)
  * Training, validation, and evaluation of emotion classifiers
  * Comparative analysis of different sampling strategies
* **Modularity:** Scripts and notebooks are reusable for other datasets and languages.

### 5. **Deployment & Architecture**

* **Telegram Integration:** The bot is built using aiogram, supporting asynchronous messaging and smooth user interaction.
* **Extensibility:** Modular codebase allows to add new emotion models, integrate other chat platforms, or expand to multi-modal inputs (e.g., speech, images).
