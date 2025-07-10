### RAG system for DT news search
## Writen and run with Python v3.11.5

import os
import re
import json
import logging
from typing import Dict, List, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import time
import textwrap


# Core LangChain imports (try different import paths for compatibility)
try:
    from langchain.chains.retrieval import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain import hub
    from langchain_community.llms import Ollama
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_community.document_loaders import TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    import logging
    from transformers.utils import logging as transformers_logging
    logging.basicConfig(level=logging.ERROR) 
    transformers_logging.set_verbosity_error() 
except ImportError:
    # Fallback for older LangChain versions
    try:
        from langchain.chains.retrieval import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain import hub
        from langchain.llms import Ollama
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.vectorstores import Chroma
        from langchain.document_loaders import TextLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document
    except ImportError:
        print("Please install LangChain and its dependencies")
        raise

# Advanced imports with fallbacks
try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer
    ADVANCED_FEATURES = True
except ImportError:
    print("Advanced features require: pip install numpy scikit-learn sentence-transformers")
    ADVANCED_FEATURES = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class RAGConfig:
    """Configuration class for RAG system parameters."""
    docs_dir: str = "./data"
    vectorstore_path: str = "./chroma_store"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "mistral"
    chunk_size: int = 1024
    chunk_overlap: int = 200
    initial_retrieval_k: int = 3
    refinement_retrieval_k: int = 15
    similarity_threshold: float = 0.7
    max_retries: int = 3
    log_level: str = "INFO"


class EnhancedRAGSystem:
    """
    RAG system with document processing, 
    semantic search, and intelligent query refinement.
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.setup_logging()
        self.setup_models()
        self.setup_llm_components()
        self.document_cache = {}
        self.text_files_content = {}  # Store content of all text files
        self.document_index = {}  # Index for fast keyword search
        
    def setup_logging(self):
        """Setup logging system."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_models(self):
        """Initialize available models."""
        self.advanced_features = ADVANCED_FEATURES
        self.spacy_available = SPACY_AVAILABLE
        self.transformers_available = TRANSFORMERS_AVAILABLE
        
        if ADVANCED_FEATURES:
            try:
                self.sentence_model = SentenceTransformer(self.config.embedding_model)
                self.logger.info("Sentence transformer model loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load sentence transformer: {e}")
                self.sentence_model = None
        else:
            self.sentence_model = None
            
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.logger.info("spaCy model loaded successfully")
            except OSError:
                self.logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
        else:
            self.nlp = None
            
        if TRANSFORMERS_AVAILABLE:
            try:
                self.ner_pipeline = pipeline("ner", model='dbmdz/bert-large-cased-finetuned-conll03-english', 
                                             aggregation_strategy="simple")
                self.logger.info("NER pipeline loaded successfully")
            except Exception as e:
                self.logger.warning(f"NER pipeline initialization failed: {e}")
                self.ner_pipeline = None
        else:
            self.ner_pipeline = None
            
    def setup_llm_components(self):
        """Setup LLM and embedding components."""
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            model_kwargs={"device": "cpu"}
        )
        
        self.llm = Ollama(
            model=self.config.llm_model,
            temperature=0.1
        )
        
    def load_and_process_documents(self) -> List[Document]:
        """Load and process documents with enhanced chunking and indexing."""
        docs_path = Path(self.config.docs_dir)
        
        if not docs_path.exists():
            raise FileNotFoundError(f"Documents directory not found: {docs_path}")
            
        all_docs = []
        supported_formats = ['.txt'] # A folder with DT news in .txt format provide via package
        
        for file_path in docs_path.rglob('*'):
            if file_path.suffix.lower() in supported_formats:
                try:
                    # Load raw content for keyword search
                    raw_content = self._load_raw_content(file_path)
                    if raw_content:
                        self.text_files_content[str(file_path)] = raw_content
                        self._build_document_index(str(file_path), raw_content)
                    
                    # Process for chunking
                    docs = self._load_document(file_path)
                    chunks = self._smart_chunking(docs, file_path)
                    all_docs.extend(chunks)
                except Exception as e:
                    self.logger.error(f"Error processing {file_path}: {e}")
                    
        return all_docs
    
    def _load_raw_content(self, file_path: Path) -> str:
        """Load raw content from file for keyword indexing."""
        try:
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return ' '.join(str(value) for value in data.values() if isinstance(value, str))
                elif isinstance(data, list):
                    return ' '.join(str(item) for item in data if isinstance(item, str))
                else:
                    return str(data)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            self.logger.error(f"Error loading raw content from {file_path}: {e}")
            return ""
    
    def _build_document_index(self, file_path: str, content: str):
        """Build an index for fast keyword search across all documents."""
        # Normalize content for better matching
        normalized_content = content.lower()
        
        # Extract words (simple tokenization)
        words = re.findall(r'\b\w+\b', normalized_content)
        
        # Build word index
        for word in words:
            if len(word) > 2:  # Skip very short words
                if word not in self.document_index:
                    self.document_index[word] = []
                if file_path not in self.document_index[word]:
                    self.document_index[word].append(file_path)
    
    def _load_document(self, file_path: Path) -> List[Document]:
        """Load document based on file type."""
        if file_path.suffix.lower() == '.json':
            return self._load_json_document(file_path)
        else:
            loader = TextLoader(str(file_path))
            return loader.load()
    
    def _load_json_document(self, file_path: Path) -> List[Document]:
        """Load JSON document and convert to Document objects."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        documents = []
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str):
                    documents.append(Document(
                        page_content=value,
                        metadata={"source": str(file_path), "key": key}
                    ))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, str):
                    documents.append(Document(
                        page_content=item,
                        metadata={"source": str(file_path), "index": i}
                    ))
        return documents
    
    def _smart_chunking(self, docs: List[Document], file_path: Path) -> List[Document]:
        """Enhanced chunking with multiple strategies."""
        if self.nlp and any(doc.page_content for doc in docs):
            return self._semantic_chunking(docs, file_path)
        else:
            return self._enhanced_traditional_chunking(docs, file_path)
    
    def _semantic_chunking(self, docs: List[Document], file_path: Path) -> List[Document]:
        """Semantic-aware chunking using NLP."""
        chunks = []
        
        for doc in docs:
            try:
                # Process with spaCy
                nlp_doc = self.nlp(doc.page_content)
                sentences = [sent.text for sent in nlp_doc.sents]
                
                current_chunk = ""
                current_size = 0
                
                for sentence in sentences:
                    sentence_size = len(sentence)
                    
                    if current_size + sentence_size > self.config.chunk_size and current_chunk:
                        chunk_doc = Document(
                            page_content=current_chunk.strip(),
                            metadata={
                                **doc.metadata,
                                "source": str(file_path),
                                "chunk_type": "semantic"
                            }
                        )
                        chunks.append(chunk_doc)
                        
                        # Start new chunk with some overlap
                        overlap_sentences = sentences[max(0, len(sentences) - 2):]
                        current_chunk = " ".join(overlap_sentences)
                        current_size = len(current_chunk)
                    else:
                        current_chunk += " " + sentence
                        current_size += sentence_size
                
                # Add final chunk
                if current_chunk.strip():
                    chunk_doc = Document(
                        page_content=current_chunk.strip(),
                        metadata={
                            **doc.metadata,
                            "source": str(file_path),
                            "chunk_type": "semantic"
                        }
                    )
                    chunks.append(chunk_doc)
                    
            except Exception as e:
                self.logger.error(f"Semantic chunking failed for {file_path}: {e}")
                # Fall back to traditional chunking
                return self._enhanced_traditional_chunking(docs, file_path)
        
        return chunks
    
    def _enhanced_traditional_chunking(self, docs: List[Document], file_path: Path) -> List[Document]:
        "Enhanced traditional chunking with better separators"
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ";", ",", " ", ""]
        )
        
        chunks = splitter.split_documents(docs)
        
        # Add enhanced metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "source": str(file_path),
                "chunk_type": "enhanced_traditional",
                "chunk_index": i
            })
        
        return chunks
    
    def create_or_load_vectorstore(self, documents: List[Document]) -> Chroma:
        """Create or load vectorstore with error handling."""
        vectorstore_path = Path(self.config.vectorstore_path)
        
        try:
            if vectorstore_path.exists():
                vectorstore = Chroma(
                    persist_directory=str(vectorstore_path),
                    embedding_function=self.embedding_model
                )
                # Test if vectorstore is working
                vectorstore.similarity_search("test", k=1)
            else:
                raise FileNotFoundError("Vectorstore not found")
                
        except Exception as e:
            self.logger.info(f"Creating new vectorstore... (Error: {e})")
            vectorstore = Chroma.from_documents(
                documents,
                self.embedding_model,
                persist_directory=str(vectorstore_path)
            )
            vectorstore.persist()
            
        return vectorstore
    
    def search_keywords_in_documents(self, keywords: List[str]) -> List[Document]:
        """Search for keywords across all loaded text files."""
        matching_docs = []
        
        for keyword in keywords:
            if len(keyword) < 2:
                continue
                
            keyword_lower = keyword.lower()
            
            # Search in document index for exact matches
            if keyword_lower in self.document_index:
                for file_path in self.document_index[keyword_lower]:
                    content = self.text_files_content.get(file_path, "")
                    if content and keyword_lower in content.lower():
                        # Extract relevant context around the keyword
                        context = self._extract_context_around_keyword(content, keyword, context_size=500)
                        matching_docs.append(Document(
                            page_content=context,
                            metadata={
                                "source": file_path,
                                "keyword": keyword,
                                "search_type": "keyword_index"
                            }
                        ))
            
            # Also search for partial matches in all documents
            for file_path, content in self.text_files_content.items():
                if keyword_lower in content.lower():
                    # Skip if already found in index search
                    if keyword_lower in self.document_index and file_path in self.document_index[keyword_lower]:
                        continue
                    
                    context = self._extract_context_around_keyword(content, keyword, context_size=500)
                    matching_docs.append(Document(
                        page_content=context,
                        metadata={
                            "source": file_path,
                            "keyword": keyword,
                            "search_type": "partial_match"
                        }
                    ))
        
        return matching_docs
    
    def _extract_context_around_keyword(self, content: str, keyword: str, context_size: int = 500) -> str:
        """Extract context around a keyword from the content."""
        keyword_lower = keyword.lower()
        content_lower = content.lower()
        
        # Find the position of the keyword
        pos = content_lower.find(keyword_lower)
        if pos == -1:
            return content[:context_size]  # Return beginning if not found
        
        # Calculate start and end positions for context
        start = max(0, pos - context_size // 2)
        end = min(len(content), pos + len(keyword) + context_size // 2)
        
        # Try to start and end at word boundaries
        while start > 0 and content[start] not in ' \n\t':
            start -= 1
        while end < len(content) and content[end] not in ' \n\t':
            end += 1
        
        context = content[start:end].strip()
        
        # Add ellipsis if we're not at the beginning/end
        if start > 0:
            context = "..." + context
        if end < len(content):
            context = context + "..."
        
        return context
    
    def extract_keywords_with_llm(self, text: str) -> Dict[str, List[str]]:
        """Extract keywords using LLM with multiple strategies."""
        try:
            # Try structured extraction first
            if self.ner_pipeline:
                entities = self.ner_pipeline(text)
                entity_words = [entity["word"] for entity in entities if len(entity["word"]) > 2]
            else:
                entity_words = []
            
            # Use spaCy if available
            if self.nlp:
                doc = self.nlp(text)
                spacy_entities = [ent.text for ent in doc.ents]
                keywords = [
                    token.lemma_.lower() for token in doc 
                    if not token.is_stop and not token.is_punct and len(token.text) > 2
                ]
            else:
                spacy_entities = []
                keywords = []
            
            # LLM-based extraction as fallback or enhancement
            llm_extraction = self._llm_keyword_extraction(text)
            
            # Combine all results
            all_entities = list(set(entity_words + spacy_entities + llm_extraction.get("entities", [])))
            all_keywords = list(set(keywords + llm_extraction.get("keywords", [])))
            
            return {
                "entities": all_entities,
                "keywords": all_keywords,
                "concepts": llm_extraction.get("concepts", [])
            }
            
        except Exception as e:
            self.logger.error(f"Keyword extraction failed: {e}")
            return {"entities": [], "keywords": [], "concepts": []}
    
    def _llm_keyword_extraction(self, text: str) -> Dict[str, List[str]]:
        """LLM-based keyword extraction with structured output."""
        try:
            prompt = f"""Extract key information from this text and return it in JSON format:
            
                    Text: {text}

                    Please extract:
                    1. entities: specific names of people, organizations, locations, products
                    2. keywords: important domain-specific terms and concepts
                    3. concepts: abstract ideas and themes

                    Return only valid JSON in this format:
                    {{"entities": ["example1", "example2"], "keywords": ["keyword1", "keyword2"], "concepts": ["concept1", "concept2"]}}

                    Example:
                    {{"entities": ["Deutsche Telekom", "AI", "CEO"], "keywords": ["investment", "artificial intelligence", "technology"], "concepts": ["digital transformation", "innovation"]}}
                    """
            
            response = self.llm.invoke(prompt)
            
            # Try to parse JSON from response
            try:
                # Look for JSON in the response
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    result = json.loads(json_str)
                    return result
                else:
                    return {"entities": [], "keywords": [], "concepts": []}
                    
            except json.JSONDecodeError:
                # If JSON parsing fails, try simple extraction
                return self._simple_keyword_extraction(text)
                
        except Exception as e:
            self.logger.error(f"LLM keyword extraction failed: {e}")
            return {"entities": [], "keywords": [], "concepts": []}
    
    def _simple_keyword_extraction(self, text: str) -> Dict[str, List[str]]:
        """Simple keyword extraction fallback."""
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        keywords = []
        
        # Extract potential keywords based on capitalization and length
        for word in words:
            if len(word) > 2 and word.lower() not in ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']:
                keywords.append(word.lower())
        
        return {
            "entities": list(set(words)),
            "keywords": list(set(keywords)),
            "concepts": []
        }
    
    def semantic_search(self, query: str, documents: List[Document], top_k: int = 10) -> List[Document]:
        """Semantic search with fallback to simple similarity."""
        if not documents:
            return []
        
        if self.sentence_model:
            try:
                # Use sentence transformer for semantic search
                query_embedding = self.sentence_model.encode([query])
                doc_texts = [doc.page_content for doc in documents]
                doc_embeddings = self.sentence_model.encode(doc_texts)
                
                similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
                
                # Create scored documents
                scored_docs = []
                for i, doc in enumerate(documents):
                    score = similarities[i]
                    if score >= self.config.similarity_threshold:
                        scored_docs.append((doc, score))
                
                # Sort by score and return top_k
                scored_docs.sort(key=lambda x: x[1], reverse=True)
                return [doc for doc, _ in scored_docs[:top_k]]
            except Exception as e:
                self.logger.warning(f"Semantic search failed: {e}")
        
        # Fallback to simple text matching
        return self._simple_text_search(query, documents, top_k)
    
    def _simple_text_search(self, query: str, documents: List[Document], top_k: int) -> List[Document]:
        "Simple text-based search as fallback"
        query_words = set(query.lower().split())
        scored_docs = []
        
        for doc in documents:
            doc_words = set(doc.page_content.lower().split())
            overlap = len(query_words.intersection(doc_words))
            if overlap > 0:
                score = overlap / len(query_words)
                scored_docs.append((doc, score))
        
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:top_k]]
    
    def answer_question_with_refinement(self, question: str) -> Dict[str, Union[str, List[str]]]:
        "Main method to answer questions with refinement using dynamic keyword search"
        start_time = time.time()
        self.logger.info(f"Processing question: {question}")
        
        try:
            # Stage 1: Initial retrieval and answer
            initial_answer, initial_sources = self._get_initial_answer(question)
            
            # Stage 2: Extract keywords and entities
            extracted_info = self.extract_keywords_with_llm(question)
            
            # Stage 3: Enhanced document retrieval using dynamic keyword search
            enhanced_docs = self._enhanced_document_retrieval(question, extracted_info)
            
            # Stage 4: Refine answer if additional relevant documents identified
            if enhanced_docs:
                final_answer = self._refine_answer(question, initial_answer, enhanced_docs)
            else:
                final_answer = initial_answer
            
            processing_time = time.time() - start_time
            self.logger.info(f"Question processed in {processing_time:.2f} seconds")
            
            return {
                "question": question,
                "answer": final_answer,
                "sources": list(set([doc.metadata.get("source", "Unknown") for doc in enhanced_docs + initial_sources])),
                "entities": extracted_info["entities"],
                "keywords": extracted_info["keywords"],
                "processing_time": processing_time
            }
            
        except Exception as e:
            self.logger.error(f"Error processing question: {e}")
            return {
                "question": question,
                "answer": "System encountered an error while processing the question. Please try again.",
                "sources": [],
                "entities": [],
                "keywords": [],
                "error": str(e)
            }
    
    def _get_initial_answer(self, question: str) -> Tuple[str, List[Document]]:
        "Get initial answer using standard RAG pipeline"
        try:
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.config.initial_retrieval_k}
            )
            
            prompt = hub.pull("langchain-ai/retrieval-qa-chat")
            document_chain = create_stuff_documents_chain(self.llm, prompt)
            qa_chain = create_retrieval_chain(retriever, document_chain)
            
            result = qa_chain.invoke({"input": question})
            
            return result['answer'], result.get('context', [])
        except Exception as e:
            self.logger.error(f"Initial answer retrieval failed: {e}")
            return "The request has not been processed", []
    
    def _enhanced_document_retrieval(self, question: str, extracted_info: Dict) -> List[Document]:
        "Enhanced document retrieval using multiple strategies including dynamic keyword search"
        enhanced_docs = []
        
        # Strategy 1: Semantic search if loaded above
        if hasattr(self, 'all_documents') and self.all_documents:
            semantic_results = self.semantic_search(question, self.all_documents, top_k=5)
            enhanced_docs.extend(semantic_results)
        
        # Strategy 2: Dynamic keyword search across all loaded text files
        all_search_terms = extracted_info["entities"] + extracted_info["keywords"] + extracted_info["concepts"]
        if all_search_terms:
            keyword_results = self.search_keywords_in_documents(all_search_terms)
            enhanced_docs.extend(keyword_results)
        
        # Strategy 3: Vector similarity search for each entity/keyword
        for term in all_search_terms:
            if len(term) > 2:
                try:
                    results = self.vectorstore.similarity_search(term, k=2)
                    enhanced_docs.extend(results)
                except Exception as e:
                    self.logger.warning(f"Vector search failed for '{term}': {e}")
        
        # Remove duplicates while preserving order
        unique_docs = []
        seen_content = set()
        
        for doc in enhanced_docs:
            # Create a hash based on content and source
            content_hash = hash((doc.page_content, doc.metadata.get("source", "")))
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        # Sort by relevance (keyword matches get higher priority)
        unique_docs.sort(key=lambda x: (
            x.metadata.get("search_type", "other") == "keyword_index",
            x.metadata.get("search_type", "other") == "partial_match"
        ), reverse=True)
        
        return unique_docs[:self.config.refinement_retrieval_k]
    
    def _refine_answer(self, question: str, initial_answer: str, enhanced_docs: List[Document]) -> str:
        "Refine the answer using enhanced document context"
        if not enhanced_docs:
            return initial_answer
        
        # Create context from enhanced documents
        context_parts = []
        for doc in enhanced_docs:
            if len("\n".join(context_parts)) < 3000:  # Prevent context overflow
                source = doc.metadata.get('source', 'Unknown')
                keyword = doc.metadata.get('keyword', '')
                search_type = doc.metadata.get('search_type', '')
                
                context_parts.append(f"Source: {source}")
                if keyword:
                    context_parts.append(f"Keyword match: {keyword}")
                if search_type:
                    context_parts.append(f"Search type: {search_type}")
                context_parts.append(doc.page_content)
                context_parts.append("---")
        
        context = "\n".join(context_parts)
        
        # Create refinement prompt
        refinement_prompt = f"""You are an expert assistant. Please provide a comprehensive answer to the question based on the original answer and additional context from multiple text files.

                            Original Question: {question}

                            Initial Answer: {initial_answer}

                            Additional Context from Text Files:
                            {context}

                            Please provide an improved answer that:
                            1. Incorporates relevant information from the additional context
                            2. Maintains accuracy and coherence
                            3. Provides specific details and examples where available
                            4. Answers the question directly and thoroughly
                            5. Synthesizes information from multiple sources when relevant

                            If the additional context doesn't improve the answer, you may return the original answer.

                            Improved Answer:"""
        
        try:
            refined_answer = self.llm.invoke(refinement_prompt)
            
            # Basic validation
            if refined_answer and len(refined_answer.strip()) > 20:
                return refined_answer
            else:
                return initial_answer
                
        except Exception as e:
            self.logger.error(f"Answer refinement failed: {e}")
            return initial_answer # Will return first answer
    
    def initialize_system(self):
        "Initialize the complete RAG system"
        
        # Load and process documents
        self.all_documents = self.load_and_process_documents()
        
        # Create vectorstore
        self.vectorstore = self.create_or_load_vectorstore(self.all_documents)
        
        # Log system capabilities
        capabilities = []
        if self.sentence_model:
            capabilities.append("Semantic Search")
        if self.nlp:
            capabilities.append("NLP Processing")
        if self.ner_pipeline:
            capabilities.append("Named Entity Recognition")
        if self.text_files_content:
            capabilities.append("Dynamic Keyword Search")
        
def main():
    "Main execution function"
    print("Enhanced RAG System with Dynamic Keyword Search")
    print("=" * 60)
    
    # Configuration
    config = RAGConfig(
        docs_dir="./data",  # Update this path as needed
        llm_model="mistral",
        chunk_size=512,
        chunk_overlap=50,
        initial_retrieval_k=3,
        refinement_retrieval_k=10,
        similarity_threshold=0.7,
        log_level="INFO"
    )
    
    try:
        # Initialize system
        rag_system = EnhancedRAGSystem(config)
        rag_system.initialize_system()
        
       
        while True:
            question = input("\nEnter your questions (type 'quit', 'exit', 'q' to exit): ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Session ended")
                break
                
            if not question:
                continue
                
            print("\nProcessing...")
            result = rag_system.answer_question_with_refinement(question)
            
            # Display results
            print("\n" + "=" * 60)
            print("RESULTS")
            print("=" * 60)
            print(f"Question: {result['question']}")
            answer_wrap = textwrap.fill(result['answer'], width=100)
            print(f"\nAnswer: {answer_wrap}")
            
            if result['sources']:
                print(f"\nSources:")
                for source in result['sources']:
                    print(f"  - {source}")
            
            if result['entities']:
                print(f"\nEntities found: {', '.join(result['entities'])}")
            
            if result['keywords']:
                print(f"Keywords found: {', '.join(result['keywords'])}")
            
            print(f"\nProcessing Time: {result.get('processing_time', 0):.2f} seconds")
            
            if 'error' in result:
                print(f"Error: {result['error']}")
        
    except KeyboardInterrupt:
        print("\nSystem interrupted by user.")
    except Exception as e:
        print(f"System initialization failed: {e}")
        print("Please check your configuration and dependencies.")
        print("\nRequired dependencies:")
        print("- pip install langchain langchain-community")
        print("- pip install chromadb")
        print("- pip install sentence-transformers")
        print("- pip install numpy scikit-learn")
        print("- pip install spacy")
        print("- python -m spacy download en_core_web_sm")
        print("- pip install transformers")


if __name__ == "__main__":
    main()