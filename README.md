# CADENCE: Context-Aware Dynamic E-commerce Neural Completion Engine

A sophisticated AI-powered search and recommendation system for e-commerce platforms, featuring real-time query completion, personalized product suggestions, and a Flipkart-style user interface.

## 🚀 Features

### Core ML Components
- **CADENCE Model**: Advanced neural architecture for e-commerce query understanding
- **Query Language Model**: Intelligent autocomplete and query suggestion
- **Catalog Language Model**: Product understanding and matching
- **Memory-Optimized Clustering**: Efficient text clustering using TF-IDF + MiniBatchKMeans

### Data Processing
- **Streaming Data Loading**: Handles massive datasets (Amazon QAC, Amazon Products 2023) without full downloads
- **Smart Sampling**: Processes millions of samples efficiently
- **Robust Error Handling**: Automatic fallbacks for memory constraints

### User Interface
- **Flipkart-Identical Design**: Pixel-perfect recreation of Flipkart's UI
- **React + TypeScript**: Modern frontend architecture
- **Real-time Search**: Instant autocomplete and product filtering
- **Responsive Design**: Mobile and desktop optimized

### API Backend
- **RESTful Endpoints**: `/autosuggest`, `/search`, `/engagement`
- **Real-time Processing**: Fast query completion and product matching
- **Scalable Architecture**: Designed for high-traffic e-commerce

## 📦 Project Structure

```
├── core/                   # Core ML models and processing
│   ├── cadence_model.py   # Main CADENCE neural architecture
│   ├── data_processor.py  # Data loading and preprocessing
│   └── personalization.py # User personalization logic
├── training/              # Model training pipeline
│   └── train_models.py   # Complete training workflow
├── api/                   # REST API endpoints
│   └── main.py           # FastAPI backend server
├── frontend/              # React user interface
│   ├── src/
│   │   ├── App.tsx       # Main React component
│   │   └── App.css       # Flipkart-style CSS
│   └── package.json      # Frontend dependencies
├── database/              # Database models and connections
├── config/                # Configuration settings
└── static/               # Static files and demo pages
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- Git

### Backend Setup
```bash
# Clone the repository
git clone https://github.com/Abhijit89Kumar/qawsed.git
cd qawsed

# Install Python dependencies
pip install -r requirements.txt

# Train the model (optional - pre-trained models included)
python train.py
```

### Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

### Quick Start Demo
```bash
# Run the simple backend + demo
python simple_backend.py

# Or run the full demo
python run_demo.py
```

## 🔥 Key Technical Innovations

### Memory-Efficient Processing
- **Streaming Architecture**: Process 1M+ samples without downloading full 60GB datasets
- **Smart Clustering**: Uses TF-IDF vectorization with MiniBatchKMeans for 10K sample clustering, then hash-based assignment for full dataset
- **Automatic Fallbacks**: Reduces sample size automatically if memory constraints detected

### Advanced ML Pipeline
- **HDBSCAN Clustering**: Sophisticated text clustering for pseudo-category generation
- **Embedding Integration**: Sentence transformer embeddings for semantic understanding
- **Multi-Modal Training**: Separate Query LM and Catalog LM for specialized understanding

### Production-Ready Features
- **Error Recovery**: Comprehensive exception handling and fallback mechanisms
- **Progress Monitoring**: Real-time memory usage and progress tracking
- **Scalable Design**: Handles datasets from 10K to 1M+ samples seamlessly

## 🎯 Training Performance

The system automatically adapts training parameters based on available resources:

- **1M Samples**: Full dataset with optimized clustering (default)
- **500K Samples**: First fallback for memory constraints
- **100K Samples**: Second fallback for limited resources
- **10K Samples**: Emergency mode for minimal systems

## 🌟 UI Features

### Flipkart-Style Interface
- **Exact Visual Replica**: Pixel-perfect recreation of Flipkart's design
- **Dynamic Search**: Real-time autocomplete with product suggestions
- **Advanced Filtering**: Category, price, rating, and brand filters
- **Responsive Grid**: Product cards with authentic styling

### Interactive Elements
- **Search Autocomplete**: Smart query suggestions as you type
- **Product Cards**: Hover effects and detailed product information
- **Navigation Menu**: Category browsing and user account features
- **Cart Integration**: Add to cart and wishlist functionality

## 🚀 API Endpoints

### Search & Autocomplete
```
GET /autosuggest?q={query}     # Get query suggestions
POST /search                   # Search products
POST /engagement              # Track user interactions
```

### Example Usage
```javascript
// Get autocomplete suggestions
fetch('/autosuggest?q=laptop')
  .then(res => res.json())
  .then(suggestions => console.log(suggestions));

// Search for products
fetch('/search', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ query: 'gaming laptop', filters: {} })
});
```

## 📊 Dataset Support

- **Amazon QAC**: Query Auto-Completion dataset (1M+ queries)
- **Amazon Products 2023**: Product catalog (250K+ products)
- **Streaming Support**: No full dataset downloads required
- **Custom Data**: Easy integration with your own datasets

## 🔧 Configuration

Key settings in `config/settings.py`:
- `BATCH_SIZE`: Training batch size (default: 32)
- `LEARNING_RATE`: Model learning rate (default: 0.001)
- `MAX_SAMPLES`: Maximum dataset samples to process
- `EMBEDDING_DIM`: Model embedding dimensions

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📈 Performance Metrics

- **Memory Usage**: <3GB RAM for 1M sample training
- **Training Time**: 15-30 minutes on CPU (varies by dataset size)
- **Query Speed**: <100ms average response time
- **UI Performance**: 60 FPS smooth interactions

## 🐛 Troubleshooting

### Common Issues
1. **Memory Errors**: Training automatically reduces sample size
2. **UMAP Import**: Fixed with proper import statements
3. **Frontend Build**: Ensure Node.js 16+ is installed
4. **API Connectivity**: Check backend server is running on correct port

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Amazon for providing the QAC and Products datasets
- Hugging Face for transformer models and datasets library
- The open-source ML community for excellent tools and libraries

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the code documentation

---

Built with ❤️ for the Flipkart GRID challenge. This system demonstrates advanced ML engineering, scalable data processing, and production-ready e-commerce AI. 