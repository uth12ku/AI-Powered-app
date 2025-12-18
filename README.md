# CVD Risk Scanner - AI-Powered Retinal Analysis

A Streamlit application for cardiovascular disease risk assessment through retinal image analysis.

## Features

- 🔬 **AI-Powered Analysis**: Simulates ResNet50 deep learning model for retinal analysis
- 🔥 **Heatmap Visualization**: GradCAM-style attention mapping on retinal images
- 📊 **Risk Assessment**: Comprehensive CVD risk scoring (0-100)
- 🔍 **Findings Detection**: Identifies retinal abnormalities associated with CVD
- 📷 **Camera Capture**: Take photos directly from your device
- 🛡️ **Secure Processing**: All processing happens locally

## Deployment to Streamlit Community Cloud

### Step 1: Create a GitHub Repository

1. Create a new GitHub repository
2. Upload these files to the repository:
   - `app.py` (main application)
   - `requirements.txt` (dependencies)
   - `README.md` (this file)

### Step 2: Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository
5. Set the main file path to `app.py`
6. Click "Deploy"

Your app will be live at `https://your-app-name.streamlit.app`

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Project Structure

```
streamlit_app/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Technical Details

### Simulated Analysis Pipeline

1. **Pre-processing**: Image normalization and preparation
2. **ResNet50 Inference**: Deep learning model prediction
3. **GradCAM Generation**: Attention heatmap creation
4. **Risk Calculation**: Final score computation

### Detectable Conditions

- Arteriovenous Nicking
- Copper/Silver Wiring
- Microaneurysms
- Cotton Wool Spots
- Hard Exudates
- Vessel Tortuosity
- Hemorrhages
- Optic Disc Edema

## Disclaimer

⚕️ This AI analysis is for screening purposes only and should not replace professional medical diagnosis. Always consult with a qualified healthcare provider for medical advice.

## License

MIT License - Feel free to use and modify for your projects.
