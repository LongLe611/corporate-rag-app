# 📚 Corporate Knowledge RAG System

AI-powered search and question-answering system for corporate documents in Vietnamese language. Built with Retrieval-Augmented Generation (RAG) technology to provide accurate, cited answers from your company's knowledge base.

## 🌟 Live Demo

**Try it now**: [https://longle-corporate-rag.streamlit.app](https://longle-corporate-rag.streamlit.app)

## ✨ Features

### 🔍 Intelligent Query System
- **Natural Language Search**: Ask questions in Vietnamese and get comprehensive answers
- **Source Citations**: Every answer includes references to source documents
- **Domain Filtering**: Filter by Corporate Tax, Accounting & Finance, or Corporate Law
- **Multi-Model Support**: Choose from Claude, GPT-4, Mistral, and more
- **Adjustable Retrieval**: Control how many source documents to retrieve

### 📤 Document Upload
- **Multi-Format Support**: Upload PDF and DOCX files
- **Automatic Processing**: Extracts text, chunks, and generates embeddings automatically
- **Domain Classification**: Organize documents by domain and type
- **Batch Upload**: Upload multiple documents at once
- **Large File Support**: Handles documents with 200+ pages

### 📊 Collection Statistics
- **Real-time Metrics**: View total documents and chunks in your knowledge base
- **Domain Breakdown**: See document distribution across domains
- **Document Inventory**: List all uploaded files with metadata

## 🏗️ Architecture

┌─────────────┐
│ Streamlit │ ← User Interface
│ Frontend │
└──────┬──────┘
│
┌──────▼──────┐
│ FastEmbed │ ← Text Embeddings
│ (BGE-small)│ (384 dimensions)
└──────┬──────┘
│
┌──────▼──────┐
│ Qdrant │ ← Vector Database
│ Cloud │ (2500+ vectors)
└──────┬──────┘
│
┌──────▼──────┐
│ OpenRouter │ ← LLM API
│ (Claude) │ (Answer Generation)
└─────────────┘

text

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | Streamlit | Web interface and user interaction |
| **Vector DB** | Qdrant Cloud | Stores document embeddings and metadata |
| **Embeddings** | FastEmbed (BGE-small-en-v1.5) | Converts text to 384-dim vectors |
| **LLM** | OpenRouter API (Claude 3.5 Sonnet) | Generates answers from retrieved context |
| **Document Processing** | PyMuPDF, python-docx | Extracts text from PDF/DOCX files |
| **Deployment** | Streamlit Cloud | Hosts the application |
| **Version Control** | GitHub | Source code management |

## 📋 Supported Domains

1. **Corporate Tax** 🏛️
   - Tax regulations and circulars
   - Corporate income tax guidelines
   - Tax compliance documentation

2. **Accounting & Finance** 💰
   - Accounting standards (TT99/2025, GAAP)
   - Financial reporting guidelines
   - Inventory management procedures

3. **Corporate Law & Regulation** ⚖️
   - Legal compliance documents
   - Corporate governance policies
   - Regulatory requirements

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Qdrant Cloud account
- OpenRouter API key

### Local Installation

1. **Clone the repository**
git clone https://github.com/LongLe611/corporate-rag-app.git
cd corporate-rag-app

text

2. **Create virtual environment**
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

text

3. **Install dependencies**
pip install -r requirements.txt

text

4. **Configure secrets**

Create `.streamlit/secrets.toml`:
[qdrant]
url = "your-qdrant-cluster-url"
api_key = "your-qdrant-api-key"

[openrouter]
api_key = "your-openrouter-api-key"

text

5. **Run the application**
streamlit run app.py

text

6. **Access the app**

Open your browser to `http://localhost:8501`

## 📁 Project Structure

corporate-rag-app/
├── .streamlit/
│ └── secrets.toml # API keys and configuration (not in repo)
├── app.py # Main Streamlit application
├── requirements.txt # Python dependencies
├── .gitignore # Git ignore patterns
└── README.md # This file

text

## 🔧 Configuration

### Qdrant Cloud Setup

1. Create account at [cloud.qdrant.io](https://cloud.qdrant.io)
2. Create a new cluster
3. Create collection named `corporate_documents` with:
   - Vector size: 384
   - Distance: Cosine

### OpenRouter API

1. Sign up at [openrouter.ai](https://openrouter.ai)
2. Generate API key
3. Add credits to your account
4. Configure in secrets.toml

## 📖 Usage Guide

### Querying Documents

1. Navigate to **"Query Documents"** tab
2. Enter your question in Vietnamese
3. Select domain filter (optional)
4. Choose LLM model and number of sources
5. Click **"Search"**
6. Review the answer with source citations

**Example queries**:
- "Chế độ kế toán TT99 năm 2025 quy định những gì?"
- "Các quy định về thuế TNDN là gì?"
- "Nguyên tắc kế toán hàng tồn kho như thế nào?"

### Uploading Documents

1. Go to **"Upload Documents"** tab
2. Select domain (Corporate Tax, Accounting & Finance, etc.)
3. Choose document type (policy, guideline, regulation, etc.)
4. Click **"Browse files"** or drag & drop PDF/DOCX files
5. Click **"Upload to Qdrant"**
6. Wait for processing to complete

### Viewing Statistics

1. Open **"Collection Stats"** tab
2. Click **"Refresh Stats"**
3. View total documents and distribution by domain

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FastEmbed** by Qdrant for efficient embeddings
- **Streamlit** for the amazing web framework
- **Qdrant** for the powerful vector database
- **OpenRouter** for unified LLM API access
- **Anthropic Claude** for intelligent answer generation

## 📧 Contact

**Project Maintainer**: LongLe611

**Issues**: Please report bugs and feature requests via [GitHub Issues](https://github.com/LongLe611/corporate-rag-app/issues)

## 🔮 Roadmap

- [ ] Add user authentication
- [ ] Implement document deletion feature
- [ ] Add conversation history
- [ ] Support for more file formats (Excel, PowerPoint)
- [ ] Advanced analytics dashboard
- [ ] Multi-language interface support
- [ ] API endpoints for integration
- [ ] Mobile-responsive design improvements

## 📊 System Requirements

**Minimum**:
- 2 GB RAM
- Modern web browser (Chrome, Firefox, Edge)
- Internet connection

**Recommended**:
- 4 GB+ RAM
- High-speed internet connection
- Latest browser version

## 🔒 Security Notes

- API keys are stored securely in Streamlit secrets
- Secrets are never committed to the repository
- All data transmission uses HTTPS
- Vector database access is protected with API keys

---

**Made with ❤️ using Python, Streamlit, and AI**