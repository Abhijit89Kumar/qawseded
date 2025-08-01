# 🚀 **KAGGLE GPU TRAINING SETUP GUIDE**

## **📋 COMPLETE SETUP CHECKLIST**

### **Step 1: GitHub Repository Setup**
```bash
# Option A: Create new repository on GitHub
# 1. Go to https://github.com/new
# 2. Repository name: "flipkart-grid-cadence"
# 3. Description: "Enhanced CADENCE model for e-commerce autocomplete"
# 4. Set to Public
# 5. Create repository

# Then update remote:
git remote set-url origin https://github.com/YOUR_USERNAME/flipkart-grid-cadence.git
git push -u origin main
```

### **Step 2: Kaggle Dataset Creation**
1. **Go to Kaggle Datasets**: https://www.kaggle.com/datasets
2. **Click "New Dataset"** → **"GitHub"**
3. **Enter Repository URL**: `https://github.com/YOUR_USERNAME/flipkart-grid-cadence`
4. **Dataset Settings**:
   - Title: "Flipkart GRID CADENCE Model"
   - Subtitle: "Enhanced e-commerce autocomplete with real Amazon data"
   - Visibility: Public
   - License: Apache 2.0
5. **Create Dataset**

### **Step 3: Kaggle Notebook Setup**
1. **Go to Kaggle Code**: https://www.kaggle.com/code
2. **Click "New Notebook"**
3. **Settings**:
   - Title: "Enhanced CADENCE Training on GPU"
   - Language: Python
   - **Accelerator: GPU T4 x2** ⚡
   - **Internet: On** 🌐
4. **Add Data Sources**:
   - Search and add your "Flipkart GRID CADENCE Model" dataset
   - Add "Amazon QAC Dataset" (if available on Kaggle)
   - Add "Amazon Products 2023" (if available on Kaggle)

### **Step 4: Upload Notebook**
Copy the contents of `kaggle_training_notebook.ipynb` to your Kaggle notebook.

### **Step 5: Environment Variables**
In your Kaggle notebook, set:
```python
import os
os.environ['GEMINI_API_KEY'] = 'AIzaSyDAMxtFaYpbqLb2dlHNAaFA6YLgMUVVVaI'
```

### **Step 6: Run Training**
Execute all cells in the notebook. Expected training time: **2-3 hours** on GPU.

## **🎯 EXPECTED RESULTS**

After successful training, you'll have:
- ✅ Enhanced CADENCE model (`kaggle_enhanced_cadence_production.pt`)
- ✅ Vocabulary mappings (`vocab.json`)
- ✅ Cluster mappings (`cluster_mappings.json`)
- ✅ Training logs and metrics

## **📥 DOWNLOAD TRAINED MODELS**

From Kaggle notebook:
```python
# Download the trained model
from IPython.display import FileLink
FileLink('models/kaggle_enhanced_cadence_production.pt')
```

## **🔄 INTEGRATE BACK TO LOCAL**

1. Download the trained models from Kaggle
2. Place them in your local `models/` directory
3. Update your local API to use the new model:
   ```python
   model_path = "models/kaggle_enhanced_cadence_production.pt"
   ```

## **⚡ PERFORMANCE EXPECTATIONS**

**On Kaggle GPU T4 x2**:
- Data Processing: ~30 minutes
- Model Training (5 epochs): ~2 hours
- Total Runtime: ~2.5 hours

**Model Performance**:
- Enhanced architecture with 512-dim embeddings
- Multi-task learning (query completion + intent + category)
- Real Amazon data integration
- E-commerce specific autocomplete

## **🚨 TROUBLESHOOTING**

**Issue: "Dataset not found"**
- Ensure the GitHub repository is public
- Check dataset name matches exactly
- Refresh Kaggle page

**Issue: "GPU not available"**
- Check notebook settings: Accelerator → GPU T4 x2
- Restart kernel if needed

**Issue: "Gemini API quota exceeded"**
- The API key is included in the setup
- Consider reducing synthetic data generation samples

**Issue: "Training hangs"**
- Check GPU memory usage
- Reduce batch size if needed
- Monitor Kaggle resource usage

## **📊 MONITORING PROGRESS**

The notebook includes progress bars and logging:
```python
logger.info("🚀 Starting CADENCE training on Kaggle GPU...")
logger.info("📊 Preparing training data...")
logger.info("🧠 Training enhanced CADENCE model...")
logger.info("✅ Training completed successfully!")
```

## **🎉 SUCCESS CRITERIA**

Training is successful when you see:
```
✅ Enhanced model training completed successfully!
🎉 Model saved as 'kaggle_enhanced_cadence_production'
```

---

**Ready to revolutionize e-commerce autocomplete with GPU-powered CADENCE! 🚀**