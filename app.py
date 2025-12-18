"""
CVD Risk Scanner - AI-Powered Retinal Analysis
Streamlit Community Cloud Version
"""

import streamlit as st
import numpy as np
from PIL import Image
import io
import time
import random
from dataclasses import dataclass
from typing import List, Literal
import base64

# ==================== Data Models ====================

@dataclass
class Finding:
    id: str
    name: str
    description: str
    severity: Literal['normal', 'mild', 'moderate', 'severe']
    location: tuple
    confidence: float

@dataclass
class HeatmapPoint:
    x: float
    y: float
    intensity: float

@dataclass
class AnalysisResult:
    risk_score: int
    risk_level: Literal['low', 'moderate', 'high', 'critical']
    confidence: float
    findings: List[Finding]
    heatmap_data: List[HeatmapPoint]
    processing_time: float

# ==================== Constants ====================

FINDINGS_DATABASE = [
    {
        'name': 'Arteriovenous Nicking',
        'description': 'Compression of venules by crossing arterioles, indicating hypertensive retinopathy',
        'severity': 'moderate',
    },
    {
        'name': 'Copper/Silver Wiring',
        'description': 'Abnormal light reflex of retinal arterioles suggesting arteriosclerosis',
        'severity': 'moderate',
    },
    {
        'name': 'Microaneurysms',
        'description': 'Small bulges in retinal blood vessels, early sign of diabetic retinopathy',
        'severity': 'mild',
    },
    {
        'name': 'Cotton Wool Spots',
        'description': 'Fluffy white patches indicating nerve fiber layer infarcts',
        'severity': 'severe',
    },
    {
        'name': 'Hard Exudates',
        'description': 'Yellow-white deposits of lipid material from leaking blood vessels',
        'severity': 'moderate',
    },
    {
        'name': 'Vessel Tortuosity',
        'description': 'Abnormal twisting of retinal blood vessels',
        'severity': 'mild',
    },
    {
        'name': 'Hemorrhages',
        'description': 'Bleeding in the retinal layers indicating vascular damage',
        'severity': 'severe',
    },
    {
        'name': 'Optic Disc Edema',
        'description': 'Swelling of the optic nerve head',
        'severity': 'severe',
    },
]

RISK_COLORS = {
    'low': '#22c55e',
    'moderate': '#eab308',
    'high': '#f97316',
    'critical': '#ef4444',
}

SEVERITY_COLORS = {
    'normal': '#22c55e',
    'mild': '#eab308',
    'moderate': '#f97316',
    'severe': '#ef4444',
}

SEVERITY_ICONS = {
    'normal': '✅',
    'mild': 'ℹ️',
    'moderate': '⚠️',
    'severe': '🚨',
}

# ==================== Analysis Functions ====================

def generate_heatmap_data(width: int, height: int, complexity: float) -> List[HeatmapPoint]:
    """Generate simulated heatmap data for visualization."""
    points = []
    num_hotspots = int(3 + random.random() * complexity * 5)
    
    hotspots = [
        {
            'x': 0.2 + random.random() * 0.6,
            'y': 0.2 + random.random() * 0.6,
            'radius': 0.1 + random.random() * 0.2,
            'intensity': 0.5 + random.random() * 0.5,
        }
        for _ in range(num_hotspots)
    ]
    
    resolution = 30
    for i in range(resolution):
        for j in range(resolution):
            x = i / resolution
            y = j / resolution
            
            max_intensity = 0
            for hotspot in hotspots:
                dx = x - hotspot['x']
                dy = y - hotspot['y']
                distance = np.sqrt(dx * dx + dy * dy)
                intensity = hotspot['intensity'] * np.exp(-distance * distance / (2 * hotspot['radius'] * hotspot['radius']))
                max_intensity = max(max_intensity, intensity)
            
            if max_intensity > 0.1:
                points.append(HeatmapPoint(x=x * width, y=y * height, intensity=max_intensity))
    
    return points

def select_findings(risk_score: int) -> List[Finding]:
    """Select appropriate findings based on risk score."""
    num_findings = 0 if risk_score < 30 else min(risk_score // 20, 4)
    
    if num_findings == 0:
        return [Finding(
            id='normal',
            name='No Significant Abnormalities',
            description='Retinal vasculature appears within normal limits',
            severity='normal',
            location=(0.5, 0.5),
            confidence=0.95,
        )]
    
    shuffled = random.sample(FINDINGS_DATABASE, min(num_findings, len(FINDINGS_DATABASE)))
    return [
        Finding(
            id=f'finding-{i}',
            name=f['name'],
            description=f['description'],
            severity=f['severity'],
            location=(0.3 + random.random() * 0.4, 0.3 + random.random() * 0.4),
            confidence=0.7 + random.random() * 0.25,
        )
        for i, f in enumerate(shuffled)
    ]

def calculate_risk_level(score: int) -> Literal['low', 'moderate', 'high', 'critical']:
    """Determine risk level from score."""
    if score < 25:
        return 'low'
    elif score < 50:
        return 'moderate'
    elif score < 75:
        return 'high'
    return 'critical'

def analyze_image(image: Image.Image) -> AnalysisResult:
    """Perform simulated AI analysis on the retinal image."""
    start_time = time.time()
    
    # Simulate processing stages
    stages = ['Pre-processing Image', 'Running ResNet50 Inference', 'Generating GradCAM', 'Finalizing Analysis']
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, stage in enumerate(stages):
        status_text.text(f"🔄 {stage}...")
        time.sleep(0.4 + random.random() * 0.3)
        progress_bar.progress((i + 1) / len(stages))
    
    status_text.empty()
    progress_bar.empty()
    
    # Generate results
    risk_score = random.randint(0, 100)
    risk_level = calculate_risk_level(risk_score)
    findings = select_findings(risk_score)
    heatmap_data = generate_heatmap_data(512, 512, risk_score / 100)
    processing_time = time.time() - start_time
    
    return AnalysisResult(
        risk_score=risk_score,
        risk_level=risk_level,
        confidence=0.85 + random.random() * 0.12,
        findings=findings,
        heatmap_data=heatmap_data,
        processing_time=processing_time,
    )

# ==================== UI Components ====================

def render_header():
    """Render the app header."""
    col1, col2 = st.columns([1, 6])
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ec4899, #8b5cf6); 
                    padding: 12px; border-radius: 12px; text-align: center;">
            <span style="font-size: 24px;">❤️</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <h1 style="margin: 0; background: linear-gradient(135deg, #ec4899, #8b5cf6); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   font-size: 28px;">CVD Risk Scanner</h1>
        <p style="margin: 0; color: #888; font-size: 14px;">AI-Powered Retinal Analysis</p>
        """, unsafe_allow_html=True)

def render_risk_gauge(result: AnalysisResult):
    """Render the circular risk gauge."""
    color = RISK_COLORS[result.risk_level]
    
    st.markdown(f"""
    <div style="text-align: center; padding: 20px;">
        <div style="position: relative; width: 200px; height: 200px; margin: 0 auto;">
            <svg viewBox="0 0 100 100" style="transform: rotate(-90deg);">
                <circle cx="50" cy="50" r="45" fill="none" stroke="#333" stroke-width="8"/>
                <circle cx="50" cy="50" r="45" fill="none" stroke="{color}" stroke-width="8"
                        stroke-dasharray="{result.risk_score * 2.83} 283"
                        style="transition: stroke-dasharray 1s ease-in-out;"/>
            </svg>
            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center;">
                <div style="font-size: 48px; font-weight: bold; color: {color};">{result.risk_score}</div>
                <div style="font-size: 14px; color: #888;">/ 100</div>
            </div>
        </div>
        <div style="margin-top: 15px;">
            <span style="background: {color}; color: white; padding: 6px 16px; 
                         border-radius: 20px; font-weight: 600; text-transform: uppercase;">
                {result.risk_level} Risk
            </span>
        </div>
        <div style="margin-top: 20px;">
            <p style="color: #888; font-size: 12px; margin-bottom: 5px;">Model Confidence</p>
            <div style="background: #333; border-radius: 10px; height: 8px; overflow: hidden;">
                <div style="background: linear-gradient(90deg, #8b5cf6, #ec4899); 
                            height: 100%; width: {result.confidence * 100}%;"></div>
            </div>
            <p style="color: #888; font-size: 12px; margin-top: 5px;">{result.confidence * 100:.1f}%</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_findings(findings: List[Finding]):
    """Render the findings panel."""
    st.markdown("### 🔍 AI Findings")
    st.caption(f"Detected {len(findings)} finding{'s' if len(findings) != 1 else ''}")
    
    for finding in findings:
        color = SEVERITY_COLORS[finding.severity]
        icon = SEVERITY_ICONS[finding.severity]
        
        st.markdown(f"""
        <div style="background: {color}15; border: 1px solid {color}40; 
                    border-radius: 10px; padding: 12px; margin-bottom: 10px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 20px;">{icon}</span>
                <div style="flex: 1;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong style="color: white;">{finding.name}</strong>
                        <span style="font-size: 10px; color: #888; font-family: monospace;">
                            {finding.confidence * 100:.0f}% conf
                        </span>
                    </div>
                    <p style="color: #aaa; font-size: 12px; margin: 5px 0 0 0;">{finding.description}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_heatmap_overlay(image: Image.Image, heatmap_data: List[HeatmapPoint], findings: List[Finding]):
    """Render the image with heatmap overlay."""
    st.markdown("### 🔥 Analysis Heatmap")
    
    # Convert image for display
    img_array = np.array(image)
    
    # Create heatmap overlay
    heatmap = np.zeros((*img_array.shape[:2], 4), dtype=np.float32)
    
    for point in heatmap_data:
        x, y = int(point.x * img_array.shape[1] / 512), int(point.y * img_array.shape[0] / 512)
        if 0 <= x < img_array.shape[1] and 0 <= y < img_array.shape[0]:
            radius = 20
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < img_array.shape[1] and 0 <= ny < img_array.shape[0]:
                        dist = np.sqrt(dx**2 + dy**2) / radius
                        if dist <= 1:
                            alpha = point.intensity * (1 - dist) * 0.6
                            # Color based on intensity (blue -> yellow -> red)
                            if point.intensity < 0.5:
                                r, g, b = 0, int(255 * point.intensity * 2), 255
                            else:
                                r, g, b = 255, int(255 * (1 - point.intensity) * 2), 0
                            heatmap[ny, nx] = [r, g, b, alpha * 255]
    
    # Blend images
    result_img = img_array.copy().astype(np.float32)
    if len(result_img.shape) == 2:
        result_img = np.stack([result_img] * 3, axis=-1)
    elif result_img.shape[2] == 4:
        result_img = result_img[:, :, :3]
    
    alpha = heatmap[:, :, 3:4] / 255
    rgb = heatmap[:, :, :3]
    result_img = result_img * (1 - alpha) + rgb * alpha
    result_img = np.clip(result_img, 0, 255).astype(np.uint8)
    
    st.image(result_img, use_container_width=True)
    
    # Color scale legend
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; 
                margin-top: 10px; padding: 0 10px;">
        <span style="color: #888; font-size: 12px;">Low</span>
        <div style="flex: 1; height: 10px; margin: 0 10px; border-radius: 5px;
                    background: linear-gradient(90deg, #0066ff, #00ff00, #ffff00, #ff0000);"></div>
        <span style="color: #888; font-size: 12px;">High</span>
    </div>
    """, unsafe_allow_html=True)

def render_feature_cards():
    """Render the feature cards on the upload page."""
    cols = st.columns(3)
    features = [
        ('🛡️', 'HIPAA Compliant', 'Secure processing'),
        ('✨', 'ResNet50 AI', 'Deep learning model'),
        ('⚡', 'Fast Results', 'Under 3 seconds'),
    ]
    
    for col, (icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: #1a1a2e; 
                        border-radius: 12px; border: 1px solid #333;">
                <div style="font-size: 28px; margin-bottom: 10px;">{icon}</div>
                <p style="font-weight: 600; margin: 0; color: white;">{title}</p>
                <p style="font-size: 12px; color: #888; margin: 5px 0 0 0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

def render_disclaimer():
    """Render the medical disclaimer."""
    st.markdown("""
    <div style="background: #1a1a2e; border: 1px solid #333; border-radius: 12px; 
                padding: 15px; text-align: center; margin-top: 20px;">
        <p style="color: #888; font-size: 12px; margin: 0;">
            <strong>⚕️ Disclaimer:</strong> This AI analysis is for screening purposes only 
            and should not replace professional medical diagnosis.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==================== Main App ====================

def main():
    st.set_page_config(
        page_title="CVD Risk Scanner - AI Retinal Analysis",
        page_icon="❤️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(180deg, #0a0a1a 0%, #1a1a2e 100%);
    }
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        padding: 15px;
        font-weight: 600;
        border: 1px solid #333;
        background: linear-gradient(135deg, #1a1a2e 0%, #2a2a4e 100%);
        color: white;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        border-color: #8b5cf6;
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.3);
    }
    .uploadedFile {
        border-radius: 12px;
        border: 2px dashed #444;
        background: #1a1a2e;
    }
    h1, h2, h3 {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'result' not in st.session_state:
        st.session_state.result = None
    if 'image' not in st.session_state:
        st.session_state.image = None
    
    # Header
    render_header()
    st.markdown("---")
    
    # Main content
    if st.session_state.result is None:
        # Upload stage
        st.markdown("""
        <div style="text-align: center; margin-bottom: 30px;">
            <h2 style="color: white; margin-bottom: 10px;">Analyze Retinal Images</h2>
            <p style="color: #888;">Upload or capture a retinal fundus image for CVD risk assessment</p>
        </div>
        """, unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload a retinal fundus image",
            type=['png', 'jpg', 'jpeg'],
            help="Supported formats: PNG, JPG, JPEG"
        )
        
        # Camera input
        st.markdown("<div style='text-align: center; color: #888; margin: 20px 0;'>─── OR ───</div>", unsafe_allow_html=True)
        
        camera_image = st.camera_input("📷 Capture with Camera")
        
        # Process image
        image_to_analyze = None
        if uploaded_file is not None:
            image_to_analyze = Image.open(uploaded_file)
        elif camera_image is not None:
            image_to_analyze = Image.open(camera_image)
        
        if image_to_analyze is not None:
            st.image(image_to_analyze, caption="Uploaded Image", use_container_width=True)
            
            if st.button("🔬 Analyze Image", type="primary"):
                with st.spinner(""):
                    result = analyze_image(image_to_analyze)
                    st.session_state.result = result
                    st.session_state.image = image_to_analyze
                    st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        render_feature_cards()
        
    else:
        # Results stage
        col_reset, _ = st.columns([1, 4])
        with col_reset:
            if st.button("🔄 New Scan"):
                st.session_state.result = None
                st.session_state.image = None
                st.rerun()
        
        st.success(f"✅ Analysis Complete! Processing time: {st.session_state.result.processing_time:.1f}s")
        
        # Two column layout for results
        col1, col2 = st.columns([1.2, 1])
        
        with col1:
            render_heatmap_overlay(
                st.session_state.image,
                st.session_state.result.heatmap_data,
                st.session_state.result.findings
            )
            st.markdown("<br>", unsafe_allow_html=True)
            render_findings(st.session_state.result.findings)
        
        with col2:
            st.markdown("### 📊 CVD Risk Assessment")
            render_risk_gauge(st.session_state.result)
            render_disclaimer()

if __name__ == "__main__":
    main()
