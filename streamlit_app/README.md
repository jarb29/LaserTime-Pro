# Machining Time Analysis - Streamlit App

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation and Setup

1. **Navigate to the app directory**
   ```bash
   cd streamlit_app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Place your trained models**
   - Create `models/sklearn/` directory
   - Place your trained sklearn models:
     - `machining_time_model.pkl`
     - `feature_scaler.pkl`
     - `model_info.txt` (optional)

4. **Run the complete setup and launch**
   ```bash
   python setup_and_run.py
   ```

This single command will:
- ✅ Check for sklearn models
- 🔄 Convert models to ONNX format (if possible)
- ✅ Validate model conversions
- 🚀 Launch the Streamlit app

### Alternative Launch Methods

**Skip model conversion (if ONNX models already exist):**
```bash
python setup_and_run.py --skip-conversion
```

**Force model reconversion:**
```bash
python setup_and_run.py --force-conversion
```

**Only validate models (don't launch app):**
```bash
python setup_and_run.py --validate-only
```

**Direct Streamlit launch (manual):**
```bash
streamlit run app.py
```

## 📁 Project Structure

```
streamlit_app/
├── app.py                          # Main Streamlit application
├── setup_and_run.py               # Setup script + app launcher
├── requirements.txt                # Dependencies
├── README.md                       # This file
│
├── models/
│   ├── sklearn/                    # Place your sklearn models here
│   │   ├── machining_time_model.pkl
│   │   ├── feature_scaler.pkl
│   │   └── model_info.txt
│   │
│   └── onnx/                       # Auto-generated ONNX models
│       ├── machining_time_model.onnx
│       ├── feature_scaler.onnx
│       └── model_metadata.json
│
├── scripts/
│   ├── convert_to_onnx.py         # ONNX conversion logic
│   └── validate_models.py         # Model validation
│
├── pages/
│   └── 01_🔮_Prediction.py        # Advanced prediction page
│
├── services/
│   ├── onnx_model_service.py      # ONNX model operations
│   └── sklearn_model_service.py   # Fallback sklearn service
│
├── components/
│   ├── sidebar.py                 # Navigation components
│   ├── forms.py                   # Input forms
│   └── charts.py                  # Visualization components
│
├── utils/
│   └── helpers.py                 # Utility functions
│
└── config/
    └── settings.py                # Configuration settings
```

## 🎯 Features

### 🔮 Prediction Capabilities
- **Single Job Prediction**: Predict machining time for individual jobs
- **Batch Processing**: Upload CSV/Excel files or manual entry for multiple jobs
- **Real-time Results**: Instant predictions with confidence intervals
- **Export Results**: Download predictions as CSV files

### 🤖 Model Support
- **ONNX Optimization**: Automatic conversion to ONNX for faster inference
- **sklearn Fallback**: Automatic fallback to sklearn if ONNX conversion fails
- **Model Validation**: Automatic validation of converted models
- **Performance Monitoring**: Track model accuracy and performance

### 📊 Visualizations
- **Interactive Charts**: Plotly-based interactive visualizations
- **Gauge Charts**: Visual representation of prediction results
- **Batch Analysis**: Charts for analyzing batch prediction results
- **Performance Metrics**: Model performance tracking over time

### 🔧 Technical Features
- **Streamlit Caching**: Optimized model loading with caching
- **Error Handling**: Robust error handling and user feedback
- **Responsive Design**: Mobile-friendly interface
- **Modular Architecture**: Clean, maintainable code structure

## 📋 Usage Guide

### 1. Home Page
- View system status and model information
- Quick prediction demo
- System overview and capabilities

### 2. Prediction Page
- **Single Prediction**: Enter material thickness and cutting length
- **Batch Processing**: Upload files or enter multiple jobs manually
- **Advanced Options**: Material type, priority, confidence levels
- **Results Export**: Download predictions as CSV

### 3. Model Management
- Upload new models
- Convert models to ONNX
- Monitor model performance
- Validate model accuracy

## 🔧 Configuration

### Model Requirements
Your sklearn models should be trained with these features:
1. `Espesor` (material thickness)
2. `Longitude Corte (m)` (cutting length)
3. `Cutting_Length_Squared` (cutting length squared)
4. `Espesor_Squared` (thickness squared)
5. `Length_Thickness_Interaction` (length × thickness)

### Environment Variables
You can set these environment variables for customization:
- `MODEL_DIR`: Custom model directory path
- `STREAMLIT_PORT`: Custom port for Streamlit (default: 8501)

## 🚨 Troubleshooting

### Common Issues

**"No model service available"**
- Ensure your sklearn models are in `models/sklearn/`
- Check that model files are not corrupted
- Try running `python setup_and_run.py --force-conversion`

**"ONNX conversion failed"**
- This is normal - the app will use sklearn fallback
- Ensure you have the latest versions of sklearn and onnx libraries
- Some complex models may not convert to ONNX

**"Module not found" errors**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+ required)

**Performance issues**
- ONNX models provide better performance
- Ensure models are cached properly (restart app if needed)
- Check available system memory

### Getting Help

1. Check the console output for detailed error messages
2. Ensure all required files are in the correct locations
3. Try the validation-only mode: `python setup_and_run.py --validate-only`
4. Check the Streamlit logs for additional debugging information

## 🔄 Development

### Adding New Features
1. Create new components in `components/`
2. Add new pages in `pages/`
3. Extend services in `services/`
4. Update configuration in `config/settings.py`

### Testing
- Use the validation script to test model conversions
- Test with sample data before using production models
- Monitor performance with different model types

## 📈 Performance

### Expected Performance
- **ONNX Models**: ~1-5ms per prediction
- **sklearn Models**: ~10-50ms per prediction
- **Batch Processing**: Optimized for large datasets
- **Memory Usage**: ~50-200MB depending on model size

### Optimization Tips
- Use ONNX models for better performance
- Enable Streamlit caching for faster loading
- Process large batches in chunks if needed
- Monitor system resources during heavy usage

## 🎉 Success!

Once everything is set up, you should see:
- ✅ Models loaded successfully
- 🌟 Streamlit app running at http://localhost:8501
- 🎯 Ready to make predictions!

Enjoy using the Machining Time Analysis system! 🚀