# LaserTime Pro
## AI-Powered Laser Cutting Time Estimator

> *Precision Timing for Precision Cutting*

LaserTime Pro is an intelligent web application that predicts laser cutting machining times with high accuracy using advanced machine learning algorithms. Built specifically for manufacturing professionals, it provides instant time estimates based on material thickness and cutting length parameters.

## ✨ Key Features

- **🚀 Instant Predictions**: Get machining time estimates in seconds
- **🎯 High Accuracy**: AI models trained on real production data  
- **📊 Dual Processing**: Individual job analysis and batch processing
- **✅ Smart Validation**: Built-in validation to ensure realistic estimates
- **💼 Professional Interface**: Clean, intuitive design for manufacturing environments
- **📥 Export Capabilities**: Download batch results for planning and scheduling
- **🔄 Auto-Training**: Models automatically retrain for optimal performance
- **⚡ ONNX Support**: High-performance inference with ONNX runtime

## 🎯 Target Users

- Manufacturing Engineers
- Production Planners  
- CNC Operators
- Project Managers
- Cost Estimators

## 📊 Use Cases

- **Production Planning**: Estimate job completion times
- **Cost Estimation**: Calculate labor and machine costs
- **Scheduling Optimization**: Plan machine utilization
- **Quote Generation**: Provide accurate customer quotes
- **Capacity Planning**: Determine production capacity

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation & Launch

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd JKI
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the app:**
   ```bash
   python deploy.py
   ```
   
   Or manually:
   ```bash
   cd streamlit_app
   python setup_and_run.py
   ```

4. **Access the app:**
   Open your browser to `http://localhost:8501`

### 📊 What happens automatically:
- Models are trained from your data
- ONNX conversion for optimal performance  
- Model validation and testing
- Streamlit app launches ready to use

## 📱 How to Use LaserTime Pro

### Individual Job Estimation
1. Enter **Material Thickness** (mm)
2. Enter **Cutting Length** (m)  
3. Click **Calculate Estimate**
4. Get instant results with confidence intervals

### Batch Processing
1. Upload CSV/Excel file with columns: `Espesor`, `Cutting_Length`
2. Or manually enter multiple jobs in the data editor
3. Click **Process Batch**
4. Download results for planning and scheduling

## 🏢 Project Structure

```
LaserTime-Pro/
├── streamlit_app/           # 📱 Main Streamlit Application
│   ├── app.py               # Main app interface
│   ├── setup_and_run.py    # Setup and deployment script
│   ├── services/            # AI model services
│   ├── components/          # UI components
│   └── config/              # Configuration
│
├── notebooks/               # 📊 Analysis & Training
│   └── analysis/
│       ├── saved_models/    # 🤖 Trained AI models
│       └── FullProgramTime.ipynb
│
├── src/                     # 🔧 Core Analysis Tools
│   ├── data/                # Data processing
│   ├── visualization/       # Plotting and dashboards
│   └── analysis/            # Statistical analysis
│
└── deploy.py                # 🚀 One-click deployment
```

## 🤖 AI Model Architecture

LaserTime Pro uses advanced machine learning with:

- **Feature Engineering**: Cutting length squared, thickness squared, interaction terms
- **Model Selection**: Automatic comparison between Random Forest and Linear Regression
- **Validation**: Built-in prediction validation and confidence intervals
- **Performance**: ONNX runtime for production-grade inference speed
- **Auto-Retraining**: Models automatically retrain to maintain accuracy

## 🔧 Technical Features

- **Streamlit Framework**: Modern, responsive web interface
- **Dual Model Support**: sklearn and ONNX for flexibility and performance
- **Smart Caching**: Efficient model loading and prediction caching
- **Error Handling**: Robust error handling and user feedback
- **Export Functions**: CSV download for batch results
- **Professional UI**: Clean, manufacturing-focused design

## 📈 Model Performance

- **High Accuracy**: R² scores typically > 0.95
- **Fast Inference**: Sub-second prediction times
- **Reliable Estimates**: Built-in validation prevents unrealistic predictions
- **Continuous Learning**: Models improve with more data

## 🛠️ Development

### For Developers

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Start development server
streamlit run streamlit_app/app.py
```

### Model Training

```bash
# Train new models
cd notebooks/analysis
python train_model.py
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## 📞 Support

For support and questions, please contact the development team or create an issue in the repository.

---

**LaserTime Pro** - *Transforming laser cutting workflow with AI-powered precision*