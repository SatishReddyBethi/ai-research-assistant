## AI Research Assistant
This project is an end-to-end Retrieval-Augmented Generation (RAG) pipeline that functions as a sophisticated Q&A agent. It is designed to work with a specialized corpus of 5 peer-reviewed IEEE publications, delivering accurate and context-aware answers to complex technical questions.

### Key Features
- **Retrieval-Augmented Generation (RAG):** The core of the project is a RAG pipeline built with LangChain. This allows the model to retrieve relevant information from a specialized database of research papers before generating a response, ensuring that the answers are both accurate and contextually appropriate.
- **Fine-Tuned Gemma-2B Model:** We have fine-tuned the Gemma-2B model on a synthetically generated dataset of academic summaries. This specialization enhances the model's ability to produce high-quality, domain-specific summaries that adhere to a formal, scientific tone.
- **Interactive Web Demo:** The application is deployed as an interactive web demo using Streamlit, providing a user-friendly interface for asking questions and receiving answers. This showcases the complete workflow, from model development to a fully functional production-ready application.
- **Containerized with Docker:** The entire service is containerized using Docker, which simplifies deployment and ensures that the application runs consistently across different environments.

### Tech Stack
- **Python:** The primary programming language for the project.
- **LangChain:** Used to build the RAG pipeline and orchestrate the different components of the system.
- **PyTorch:** The deep learning framework used for fine-tuning the Gemma-2B model.
- **Hugging Face Transformers:** Provides the tools and pre-trained models needed for fine-tuning and inference.
- **ChromaDB:** A vector database used to store and retrieve document embeddings for the RAG pipeline.
- **Streamlit:** Used to create the interactive web demo.
- **Docker:** Used to containerize the application for easy deployment.

### Project Structure
```
.
├── .gitignore
├── .dockerignore
├── .env-example                    # Create a .env file using this
├── data                            # Research Papers in PDF format
├── Dockerfile
├── app.py                          # Stremlit app
├── fine_tuner.py
├── generate_synthetic_data.py      # Generate Synthetic Dataset for Fine-tuning
├── hugging_face_model_upload.py    # Upload model to hugging face
├── inference.py
├── q_and_a_rag_model.py
├── utils.py
├── LICENSE
├── Makefile
├── requirements.txt
└── README.md
```

### Setup and Installation
1. **Clone the repository:**
    ```
    git clone [https://github.com/your-username/ai-research-assistant.git](https://github.com/your-username/ai-research-assistant.git)
    cd ai-research-assistant
    ```
2. **Create and activate a virtual environment:**
    ```
    python3 -m venv venv
    source venv/bin/activate
    ```
3. **Install the required dependencies:**
    ```
    pip install -r requirements.txt
    ```
    Note: If you are using Nvidia GPU or Intel Arc GPU (XPU), you will have to install alternative versions of torch to train/run on your GPU/XPU

### Usage
1. **Prepare the data:** Run the generate_synthetic_data.py script to process the IEEE publications and create the synthetic dataset for fine-tuning.
    ```
    python generate_synthetic_data.py
    ```
2. **Fine-tune the model:** Run the fine_tuner.py script to fine-tune the Gemma-2B model on the prepared dataset.
    ```
    python fine_tuner.py
    ```
3. **Run the Streamlit application:** Start the interactive web demo by running the app.py script.
    ```
    streamlit run app.py
    ```
    Then, go to:
    ```
    http://localhost:8501
    ```
4. **Build and run with Docker:** To build and run the application using Docker, use the following commands:
    ```
    docker build -t ai-research-assistant .
    docker run -p 8501:8501 ai-research-assistant
    ```
    Then, go to:
    ```
    http://localhost:8501
    ```

### How It Works

1. **Data Preparation:** The prepare_data.py script loads the 5 peer-reviewed IEEE publications, processes them, and generates a synthetic dataset of academic summaries. This dataset is crucial for fine-tuning the Gemma-2B model to the specific domain.

2. **Fine-Tuning:** The fine_tune.py script takes the synthetic dataset and fine-tunes the Gemma-2B model. This process adapts the model to the nuances of academic writing, improving its summarization capabilities and ensuring a formal, scientific tone.

3. **RAG Pipeline:** The app.py script implements the RAG pipeline. When a user asks a question, the system first retrieves relevant text chunks from the research papers stored in ChromaDB. These chunks are then passed to the fine-tuned Gemma-2B model, which generates a context-aware and accurate answer.

4. **Deployment:** The entire application is containerized with Docker, making it easy to deploy and scale. The Streamlit front-end provides a simple and intuitive interface for users to interact with the Q&A agent.

This project demonstrates a complete workflow, from data preparation and model fine-tuning to building a RAG pipeline and deploying a production-ready application. It showcases how to create a powerful Q&A agent that can handle complex technical queries in a specialized domain.