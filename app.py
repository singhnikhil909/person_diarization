"""
Streamlit App for Speaker Identification
"""

import streamlit as st
import os
import tempfile
import numpy as np
from speaker_identifier import SpeakerIdentifier
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Speaker Identification System",
    page_icon="🎤",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div {
        background-color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'identifier' not in st.session_state:
    st.session_state.identifier = None
if 'reference_voices_loaded' not in st.session_state:
    st.session_state.reference_voices_loaded = False
if 'model_type' not in st.session_state:
    st.session_state.model_type = 'resemblyzer'

# Header
st.markdown('<h1 class="main-header">🎤 Speaker Identification System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    model_type = st.selectbox(
        "Model Type",
        options=["resemblyzer", "pyannote"],
        index=0,
        help="Choose between Resemblyzer or Pyannote Community 3.1 model"
    )
    
    if model_type == "pyannote":
        st.info("📝 **Pyannote Setup:**\n\n"
                "1. Accept user agreement at [pyannote/embedding](https://huggingface.co/pyannote/embedding)\n"
                "2. Set HF_TOKEN environment variable with your Hugging Face token\n"
                "3. Or login via: `huggingface-cli login`")
    
    voice_samples_dir = st.text_input(
        "Voice Samples Directory",
        value="voice_samples",
        help="Directory containing reference voice samples"
    )
    
    segment_duration = st.slider(
        "Segment Duration (seconds)",
        min_value=2.0,
        max_value=10.0,
        value=3.0,
        step=0.5,
        help="Duration of each audio segment for analysis"
    )
    
    overlap = st.slider(
        "Segment Overlap (seconds)",
        min_value=0.0,
        max_value=2.0,
        value=1.0,
        step=0.5,
        help="Overlap between segments"
    )
    
    similarity_threshold = st.slider(
        "Similarity Threshold (%)",
        min_value=0,
        max_value=100,
        value=50,
        help="Minimum similarity percentage to consider a match"
    )
    
    if st.button("🔄 Load Reference Voices", type="primary"):
        with st.spinner(f"Loading reference voices with {model_type} model..."):
            try:
                identifier = SpeakerIdentifier(voice_samples_dir, model_type=model_type)
                identifier.load_reference_voices()
                st.session_state.identifier = identifier
                st.session_state.reference_voices_loaded = True
                st.session_state.model_type = model_type
                st.success(f"✅ Loaded {len(identifier.speaker_names)} speakers with {model_type.upper()} model!")
                st.json(identifier.speaker_names)
            except Exception as e:
                st.error(f"Error loading reference voices: {e}")
                st.session_state.reference_voices_loaded = False
                import traceback
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📤 Upload Audio File")
    
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['mp3', 'wav', 'm4a', 'mp4', 'flac', 'ogg'],
        help="Upload an audio file containing voices to identify"
    )

with col2:
    st.header("📊 Reference Speakers")
    if st.session_state.reference_voices_loaded and st.session_state.identifier:
        model_name = st.session_state.get('model_type', 'resemblyzer').upper()
        st.success(f"✅ {len(st.session_state.identifier.speaker_names)} speakers loaded")
        st.info(f"🤖 Model: **{model_name}**")
        for speaker in st.session_state.identifier.speaker_names:
            st.write(f"• {speaker}")
    else:
        st.warning("⚠️ Please load reference voices first")

st.markdown("---")

# Process uploaded audio
if uploaded_file is not None:
    if not st.session_state.reference_voices_loaded:
        st.error("❌ Please load reference voices first using the sidebar!")
    else:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            st.header("🔍 Analysis Results")
            
            with st.spinner("Analyzing audio... This may take a few moments..."):
                identifier = st.session_state.identifier
                
                # Get overall matches
                overall_matches = identifier.get_overall_speaker_matches(tmp_path)
                
                if overall_matches:
                    # Display results in columns
                    st.subheader("📈 Overall Speaker Matches")
                    
                    # Create metrics for top matches
                    top_matches = overall_matches[:5]  # Top 5 matches
                    
                    cols = st.columns(min(len(top_matches), 5))
                    for idx, match in enumerate(top_matches):
                        with cols[idx % 5]:
                            percentage = match['average_percentage']
                            st.metric(
                                label=match['speaker'],
                                value=f"{percentage:.1f}%",
                                delta=f"{match['segments_found']} segments"
                            )
                    
                    # Detailed results table
                    st.subheader("📋 Detailed Results")
                    
                    # Filter by threshold
                    filtered_matches = [
                        m for m in overall_matches 
                        if m['average_percentage'] >= similarity_threshold
                    ]
                    
                    if filtered_matches:
                        # Create DataFrame-like display
                        results_data = []
                        for match in filtered_matches:
                            results_data.append({
                                'Speaker': match['speaker'],
                                'Average Similarity (%)': f"{match['average_percentage']:.2f}",
                                'Max Similarity (%)': f"{match['max_percentage']:.2f}",
                                'Segments Found': match['segments_found']
                            })
                        
                        # Display as table
                        import pandas as pd
                        df = pd.DataFrame(results_data)
                        st.dataframe(df, use_container_width=True)
                        
                        # Visualization
                        st.subheader("📊 Similarity Visualization")
                        
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                        
                        # Bar chart for average similarity
                        speakers = [m['speaker'] for m in filtered_matches]
                        avg_scores = [m['average_percentage'] for m in filtered_matches]
                        
                        bars1 = ax1.barh(speakers, avg_scores, color='steelblue')
                        ax1.set_xlabel('Average Similarity (%)')
                        ax1.set_title('Average Similarity Scores')
                        ax1.set_xlim(0, 100)
                        ax1.grid(axis='x', alpha=0.3)
                        
                        # Add value labels on bars
                        for i, (bar, score) in enumerate(zip(bars1, avg_scores)):
                            ax1.text(score + 1, i, f'{score:.1f}%', 
                                    va='center', fontweight='bold')
                        
                        # Bar chart for max similarity
                        max_scores = [m['max_percentage'] for m in filtered_matches]
                        bars2 = ax2.barh(speakers, max_scores, color='coral')
                        ax2.set_xlabel('Max Similarity (%)')
                        ax2.set_title('Maximum Similarity Scores')
                        ax2.set_xlim(0, 100)
                        ax2.grid(axis='x', alpha=0.3)
                        
                        # Add value labels on bars
                        for i, (bar, score) in enumerate(zip(bars2, max_scores)):
                            ax2.text(score + 1, i, f'{score:.1f}%', 
                                    va='center', fontweight='bold')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Progress bars for each speaker
                        st.subheader("🎯 Similarity Breakdown")
                        for match in filtered_matches:
                            col_a, col_b = st.columns([2, 1])
                            with col_a:
                                st.write(f"**{match['speaker']}**")
                                st.progress(match['average_percentage'] / 100)
                            with col_b:
                                st.metric(
                                    "Match",
                                    f"{match['average_percentage']:.1f}%",
                                    f"Max: {match['max_percentage']:.1f}%"
                                )
                        
                        # Summary
                        st.subheader("✅ Summary")
                        detected_speakers = [
                            m['speaker'] for m in filtered_matches 
                            if m['average_percentage'] >= similarity_threshold
                        ]
                        
                        if detected_speakers:
                            st.success(f"🎉 **Detected Speakers:** {', '.join(detected_speakers)}")
                        else:
                            st.info("ℹ️ No speakers detected above the threshold")
                    else:
                        st.warning(f"⚠️ No speakers found with similarity >= {similarity_threshold}%")
                        
                        # Show all matches anyway
                        st.info("Showing all matches below threshold:")
                        for match in overall_matches[:3]:
                            st.write(f"- **{match['speaker']}**: {match['average_percentage']:.1f}%")
                else:
                    st.error("❌ Could not analyze audio. Please check the audio file format.")
                    
        except Exception as e:
            st.error(f"❌ Error processing audio: {e}")
            st.exception(e)
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass

else:
    # Instructions
    st.info("""
    👋 **Welcome to the Speaker Identification System!**
    
    **How to use:**
    1. Click "Load Reference Voices" in the sidebar to load voice samples
    2. Upload an audio file containing voices to identify
    3. Adjust settings in the sidebar (segment duration, overlap, threshold)
    4. View the results showing which speakers are detected
    
    **Supported audio formats:** MP3, WAV, M4A, MP4, FLAC, OGG
    """)

# Footer
st.markdown("---")
model_info = st.session_state.get('model_type', 'resemblyzer').upper()
st.markdown(
    f"<div style='text-align: center; color: #666;'>Built with {model_info}, Resemblyzer, Pyannote & Streamlit</div>",
    unsafe_allow_html=True
)

