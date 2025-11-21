import time
import streamlit as st
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
import httpx

# Page configuration
st.set_page_config(
    page_title="Corporate Knowledge RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize connections with caching
@st.cache_resource(show_spinner="Initializing clients...")
def init_clients():
    """Initialize Qdrant, FastEmbed, and HTTP client"""
    try:
        # Initialize Qdrant client
        qdrant_client = QdrantClient(
            url=st.secrets["QDRANT_URL"],
            api_key=st.secrets["QDRANT_API_KEY"]
        )
        
        # Initialize FastEmbed (local, no API needed)
        embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        # Initialize HTTP client for OpenRouter
        http_client = httpx.Client(timeout=30.0)
        
        return qdrant_client, embedding_model, http_client
    except Exception as e:
        st.error(f"Error initializing clients: {e}")
        st.stop()

# Load clients
qdrant_client, embedding_model, http_client = init_clients()

def generate_embeddings(text):
    """Generate embeddings using FastEmbed (local)"""
    try:
        embeddings = list(embedding_model.embed([text]))
        return embeddings[0]
    except Exception as e:
        st.error(f"Embedding error: {e}")
        raise

def call_openrouter_llm(prompt, model="anthropic/claude-sonnet-4.5"):
    """Call OpenRouter API directly using httpx"""
    try:
        response = http_client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {st.secrets['OPENROUTER_API_KEY']}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8501"
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1500
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as e:
        st.error(f"HTTP Error: {e.response.status_code}")
        return f"Error: {e.response.status_code} - {e.response.text[:200]}"
    except Exception as e:
        st.error(f"LLM Error: {e}")
        return f"Error: {str(e)}"

def rag_query(user_question, domain_filter=None, top_k=5, model="anthropic/claude-sonnet-4.5"):
    """Execute RAG query: retrieve relevant chunks and generate answer"""
    # Generate query embedding (local, fast)
    query_embedding = generate_embeddings(user_question)
    
    # Build query parameters
    must_conditions = []
    if domain_filter and domain_filter != "All Domains":
        must_conditions.append(
            models.FieldCondition(
                key="domain",
                match=models.MatchValue(value=domain_filter)
            )
        )
    
    # Execute search
    search_results = qdrant_client.query_points(
        collection_name="corporate_documents",
        query=query_embedding.tolist(),
        limit=top_k,
        with_payload=True,
        query_filter=models.Filter(must=must_conditions) if must_conditions else None
    )
    
    # Format context and sources
    context_parts = []
    sources = []
    
    for idx, point in enumerate(search_results.points):
        payload = point.payload or {}
        text = payload.get("text", "")
        context_parts.append(
            f"[Source {idx+1}] (Relevance: {point.score:.3f})\n"
            f"Domain: {payload.get('domain', 'N/A')}\n"
            f"Document: {payload.get('source', 'N/A')}\n"
            f"Content: {text}\n"
        )
        sources.append({
            "text": text[:200] + "..." if len(text) > 200 else text,
            "domain": payload.get("domain", "N/A"),
            "source": payload.get("source", "N/A"),
            "document_type": payload.get("document_type", "N/A"),
            "relevance": point.score
        })
    
    if not sources:
        return "I couldn't find any relevant information in the knowledge base to answer your question.", []
    
    context = "\n".join(context_parts)
    
    # Create prompt
    prompt = f"""You are an expert AI assistant specializing in corporate finance, tax law, and corporate regulations.

CONTEXT FROM KNOWLEDGE BASE:
{context}

USER QUESTION: {user_question}

INSTRUCTIONS:
1. Answer the question using ONLY the information provided in the context above
2. If the context doesn't contain enough information, clearly state that
3. Cite which source(s) you used (e.g., "According to Source 1...")
4. Be precise, professional, and comprehensive
5. For legal/tax matters, remind users to consult professionals

ANSWER:"""
    
    # Generate response using direct HTTP call
    answer = call_openrouter_llm(prompt, model)
    
    return answer, sources

# Main UI
st.markdown('<div class="main-header">📚 Corporate Knowledge RAG System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered search for Corporate Finance, Tax, and Law</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Domain filter
    domain_options = ["All Domains", "Accounting & Finance", "Corporate Tax", "Corporate Law"]
    selected_domain = st.selectbox(
        "Filter by Domain",
        domain_options,
        help="Narrow your search to a specific domain"
    )
    
    # Number of sources
    num_sources = st.slider(
        "Number of sources to retrieve",
        min_value=1,
        max_value=10,
        value=5,
        help="How many relevant document chunks to use"
    )
    
    # Model selection
    model_options = {
        "Claude 4.5 Sonnet (Recommended)": "anthropic/claude-sonnet-4.5",
        "Claude 4.5 Haiku (Faster)": "anthropic/claude-haiku-4.5",
        "GPT-4o Mini (Cheap)": "openai/gpt-4o-mini",
        "Mistral Large 2": "mistralai/mistral-large-2"
    }
    selected_model_name = st.selectbox(
        "AI Model",
        list(model_options.keys()),
        help="Choose the language model for generating answers"
    )
    selected_model = model_options[selected_model_name]
    
    st.divider()
    
    # Information section
    st.subheader("📋 About")
    st.write("""
    This RAG system provides intelligent answers to questions about:
    - **Accounting & Finance**: GAAP standards, financial statements
    - **Corporate Tax**: Deductions, documentation requirements
    - **Corporate Law**: Governance, fiduciary duties
    
    **Tech Stack:**
    - Embeddings: FastEmbed (local, offline)
    - LLM: OpenRouter API (via httpx)
    - Vector DB: Qdrant Cloud
    """)
    
    st.info("💡 **Tip**: Be specific in your questions for better results!")
    
    st.divider()
    
    # Stats
    st.subheader("📊 System Info")
    try:
        collection_info = qdrant_client.get_collection("corporate_documents")
        st.metric("Documents in Knowledge Base", collection_info.points_count)
        st.metric("Vector Dimensions", collection_info.config.params.vectors.size)
    except:
        st.warning("Unable to fetch collection stats")

# Main query interface
st.subheader("🔍 Ask a Question")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Query input
user_query = st.text_area(
    "Enter your question:",
    height=100,
    placeholder="e.g., What are the required financial statements for public companies?",
    help="Ask any question about corporate finance, tax, or law"
)

col1, col2, col3 = st.columns([1, 1, 4])

with col1:
    search_button = st.button("🔍 Search", type="primary", use_container_width=True)

with col2:
    clear_button = st.button("🗑️ Clear History", use_container_width=True)

if clear_button:
    st.session_state.chat_history = []
    st.rerun()

# Process query
if search_button and user_query:
    with st.spinner("🤔 Searching knowledge base and generating answer..."):
        try:
            answer, sources = rag_query(
                user_query,
                domain_filter=selected_domain,
                top_k=num_sources,
                model=selected_model
            )
            
            # Add to chat history
            st.session_state.chat_history.insert(0, {
                "question": user_query,
                "answer": answer,
                "sources": sources,
                "domain": selected_domain,
                "model": selected_model_name
            })
            
        except Exception as e:
            st.error(f"Error processing query: {e}")

# Display results
if st.session_state.chat_history:
    st.divider()
    
    for idx, chat in enumerate(st.session_state.chat_history):
        with st.expander(f"❓ {chat['question']}", expanded=(idx == 0)):
            # Question metadata
            st.caption(f"🎯 Domain: {chat['domain']} | 🤖 Model: {chat['model']}")
            
            # Answer
            st.markdown("### 💡 Answer")
            st.markdown(chat['answer'])
            
            # Sources
            if chat['sources']:
                st.markdown("### 📚 Sources")
                for i, source in enumerate(chat['sources'], 1):
                    with st.container():
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>[{i}] {source['source']}</strong><br>
                            <small>Domain: {source['domain']} | Type: {source['document_type']} | Relevance: {source['relevance']:.3f}</small>
                            <hr style="margin: 0.5rem 0;">
                            <div style="max-height: 150px; overflow-y: auto;">
                                {source['text']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No sources found for this query.")
            
            st.divider()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>⚠️ <strong>Disclaimer:</strong> This system provides information for reference only. 
    Always consult qualified professionals for legal, tax, and financial advice.</p>
    <p>Powered by Qdrant • OpenRouter • Streamlit</p>
</div>
""", unsafe_allow_html=True)