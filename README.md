## For the following system to run please install the following dependencies

#### Core dependencies
pip install langchain
pip install langchain-community
pip install chromadb

#### Advanced features (required for full functionality)
pip install numpy
pip install scikit-learn
pip install sentence-transformers
pip install spacy
pip install transformers

#### Additional NLP processing
python -m spacy download en_core_web_sm

#### For running the Mistral model via Ollama
pip install ollama
ollama pull mistral
Install ollama localy as well https://ollama.com/download
Ollama has to be running locally

#### For Jupyter notebook support (if running the notebook)
pip install ipywidgets
pip install jupyter

#### Other utility packages
pip install tqdm
pip install textwrap
