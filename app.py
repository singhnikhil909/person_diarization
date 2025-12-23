"""
Streamlit UI for Speaker Diarization
"""

import streamlit as st
import tempfile
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import time
import warnings
from dotenv import load_dotenv
from diarization_utils import (
    load_diarization_pipeline,
    perform_diarization,
    format_time
)

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables with explicit encoding and file path
try:
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(dotenv_path=env_path, encoding='utf-8')
except Exception as e:
    st.warning(f"Could not load .env file: {e}")

# Page configuration
st.set_page_config(
    page_title="Speaker Diarization",
    page_icon="🎤",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .result-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Load and cache the diarization model"""
    # Use transformers-based model (BUT-FIT/diarizen-wavlm-large-s80-md-v2)
    return load_diarization_pipeline(model_name="BUT-FIT/diarizen-wavlm-large-s80-md-v2")


def process_diarization_simple(diarization):
    """
    Process diarization results and label speakers as Person 1, Person 2, etc.
    Merges consecutive segments from the same speaker.
    
    Args:
        diarization: Diarization results from pipeline
        
    Returns:
        List of dictionaries with speaker segments
    """
    # First, collect all segments with their speakers
    segments_list = []
    speaker_map = {}  # Maps original labels to Person 1, Person 2, etc.
    person_counter = 1
    
    for segment, track, label in diarization.itertracks(yield_label=True):
        # Map original label to Person N
        if label not in speaker_map:
            speaker_map[label] = f"Person {person_counter}"
            person_counter += 1
        
        speaker_name = speaker_map[label]
        segments_list.append({
            'start': segment.start,
            'end': segment.end,
            'speaker': speaker_name,
            'duration': segment.end - segment.start
        })
    
    # Sort by start time
    segments_list.sort(key=lambda x: x['start'])
    
    # Merge consecutive segments from the same speaker
    if not segments_list:
        return [], {}
    
    results = []
    current_segment = segments_list[0].copy()
    
    for seg in segments_list[1:]:
        # If same speaker and segments are close (gap < 0.5 seconds), merge them
        if (seg['speaker'] == current_segment['speaker'] and 
            seg['start'] - current_segment['end'] < 0.5):
            # Merge: extend the end time
            current_segment['end'] = seg['end']
            current_segment['duration'] = current_segment['end'] - current_segment['start']
        else:
            # Different speaker or gap too large - save current and start new
            results.append(current_segment)
            current_segment = seg.copy()
    
    # Add the last segment
    results.append(current_segment)
    
    return results, speaker_map


def create_timeline_visualization(results):
    """Create a timeline visualization of speaker segments"""
    if not results:
        return None
    
    # Prepare data for visualization
    speakers = sorted(set(r['speaker'] for r in results))
    colors = px.colors.qualitative.Set3[:len(speakers)]
    speaker_colors = {speaker: colors[i] for i, speaker in enumerate(speakers)}
    
    fig = go.Figure()
    
    y_positions = {speaker: i for i, speaker in enumerate(speakers)}
    
    for result in results:
        speaker = result['speaker']
        start = result['start']
        end = result['end']
        duration = result['duration']
        
        fig.add_trace(go.Scatter(
            x=[start, end, end, start, start],
            y=[y_positions[speaker] - 0.4, y_positions[speaker] - 0.4, 
               y_positions[speaker] + 0.4, y_positions[speaker] + 0.4, 
               y_positions[speaker] - 0.4],
            fill='toself',
            fillcolor=speaker_colors[speaker],
            line=dict(color=speaker_colors[speaker], width=2),
            mode='lines',
            name=speaker,
            showlegend=False,
            hovertemplate=f'<b>{speaker}</b><br>' +
                         f'Start: {format_time(start)}<br>' +
                         f'End: {format_time(end)}<br>' +
                         f'Duration: {format_time(duration)}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Speaker Timeline",
        xaxis_title="Time (seconds)",
        yaxis=dict(
            tickmode='array',
            tickvals=list(y_positions.values()),
            ticktext=list(y_positions.keys()),
            title="Speaker"
        ),
        height=400,
        hovermode='closest',
        showlegend=False
    )
    
    return fig


def create_summary_chart(results):
    """Create a pie chart showing speaking time distribution"""
    if not results:
        return None
    
    speaker_stats = {}
    for result in results:
        speaker = result['speaker']
        duration = result['duration']
        if speaker not in speaker_stats:
            speaker_stats[speaker] = 0.0
        speaker_stats[speaker] += duration
    
    speakers = list(speaker_stats.keys())
    durations = [speaker_stats[s] for s in speakers]
    
    fig = go.Figure(data=[go.Pie(
        labels=speakers,
        values=durations,
        hole=0.3,
        textinfo='label+percent',
        texttemplate='%{label}<br>%{percent}<br>(%{value:.1f}s)',
        hovertemplate='<b>%{label}</b><br>' +
                     'Duration: %{value:.1f} seconds<br>' +
                     'Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title="Speaking Time Distribution",
        height=400
    )
    
    return fig


def main():
    st.markdown('<h1 class="main-header">🎤 Speaker Diarization</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Preload models at startup (cached, so only loads once)
    # This ensures models are ready before user uploads a file
    try:
        pipeline = load_pipeline()
        # Show status in sidebar
        with st.sidebar:
            st.success("✅ Models loaded and ready!")
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.info("💡 Make sure you've accepted the terms of use at:\n- https://huggingface.co/pyannote/speaker-diarization-3.1\n- https://huggingface.co/pyannote/embedding")
        st.stop()
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        min_speakers = st.number_input(
            "Minimum Speakers",
            min_value=1,
            max_value=20,
            value=2,
            help="Set minimum number of speakers (default: 2)"
        )
        max_speakers = st.number_input(
            "Maximum Speakers",
            min_value=1,
            max_value=20,
            value=2,
            help="Set maximum number of speakers (default: 2)"
        )
        st.markdown("---")
        st.info("💡 Upload an audio file to start diarization. Speakers will be labeled as Person 1, Person 2, etc.")
    
    # File uploader
    st.header("📁 Upload Audio File")
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'flac', 'm4a', 'ogg'],
        help="Supported formats: WAV, MP3, FLAC, M4A, OGG"
    )
    
    if uploaded_file is not None:
        # Display file info
        file_details = {
            "Filename": uploaded_file.name,
            "FileType": uploaded_file.type,
            "FileSize": f"{uploaded_file.size / (1024*1024):.2f} MB"
        }
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Filename", file_details["Filename"])
        with col2:
            st.metric("Type", file_details["FileType"])
        with col3:
            st.metric("Size", file_details["FileSize"])
        
        # Process button
        if st.button("🚀 Start Diarization", type="primary", use_container_width=True):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Initialize progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Get preloaded pipeline (already loaded at startup)
                pipeline = load_pipeline()
                progress_bar.progress(20)
                
                # Perform diarization
                status_text.text("Analyzing audio and identifying speakers...")
                progress_bar.progress(30)
                
                # Create a placeholder for progress updates
                progress_placeholder = st.empty()
                
                diarization = perform_diarization(
                    tmp_path,
                    pipeline,
                    min_speakers=min_speakers if min_speakers else None,
                    max_speakers=max_speakers if max_speakers else None
                )
                progress_bar.progress(80)
                progress_placeholder.empty()  # Clear progress updates
                
                # Process results
                status_text.text("Processing results...")
                results, speaker_map = process_diarization_simple(diarization)
                progress_bar.progress(100)
                
                status_text.text("✓ Diarization complete!")
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                # Display results
                st.markdown("---")
                st.header("📊 Results")
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                
                total_duration = sum(r['duration'] for r in results)
                num_speakers = len(speaker_map)
                num_segments = len(results)
                
                with col1:
                    st.metric("Total Duration", format_time(total_duration))
                with col2:
                    st.metric("Number of Speakers", num_speakers)
                with col3:
                    st.metric("Number of Segments", num_segments)
                with col4:
                    avg_segment = total_duration / num_segments if num_segments > 0 else 0
                    st.metric("Avg Segment Length", format_time(avg_segment))
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    timeline_fig = create_timeline_visualization(results)
                    if timeline_fig:
                        st.plotly_chart(timeline_fig, use_container_width=True)
                
                with col2:
                    summary_fig = create_summary_chart(results)
                    if summary_fig:
                        st.plotly_chart(summary_fig, use_container_width=True)
                
                # Detailed results table
                st.subheader("📋 Detailed Timeline")
                
                # Prepare data for table
                table_data = []
                for result in results:
                    table_data.append({
                        'Start Time': format_time(result['start']),
                        'End Time': format_time(result['end']),
                        'Duration': format_time(result['duration']),
                        'Speaker': result['speaker']
                    })
                
                df = pd.DataFrame(table_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Speaker statistics
                st.subheader("📈 Speaker Statistics")
                speaker_stats = {}
                for result in results:
                    speaker = result['speaker']
                    duration = result['duration']
                    if speaker not in speaker_stats:
                        speaker_stats[speaker] = {'total_time': 0.0, 'segments': 0}
                    speaker_stats[speaker]['total_time'] += duration
                    speaker_stats[speaker]['segments'] += 1
                
                stats_data = []
                for speaker, stats in sorted(speaker_stats.items(), key=lambda x: x[1]['total_time'], reverse=True):
                    percentage = (stats['total_time'] / total_duration) * 100
                    stats_data.append({
                        'Speaker': speaker,
                        'Total Speaking Time': format_time(stats['total_time']),
                        'Percentage': f"{percentage:.1f}%",
                        'Number of Segments': stats['segments'],
                        'Avg Segment Length': format_time(stats['total_time'] / stats['segments'])
                    })
                
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                # Download results
                st.subheader("💾 Download Results")
                results_json = pd.DataFrame(results).to_json(orient='records', indent=2)
                st.download_button(
                    label="📥 Download Results as JSON",
                    data=results_json,
                    file_name=f"diarization_results_{uploaded_file.name}.json",
                    mime="application/json"
                )
                
                # CSV download
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"diarization_results_{uploaded_file.name}.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"❌ Error during diarization: {str(e)}")
                st.exception(e)
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
    
    else:
        # Show instructions when no file is uploaded
        st.info("👆 Please upload an audio file to begin speaker diarization.")
        
        with st.expander("ℹ️ How to use"):
            st.markdown("""
            **Steps:**
            1. Upload an audio file using the file uploader above
            2. (Optional) Set minimum and maximum speaker counts in the sidebar
            3. Click "Start Diarization" to process the audio
            4. View the results including:
               - Timeline visualization
               - Speaking time distribution
               - Detailed segment information
               - Speaker statistics
            5. Download results as JSON or CSV
            
            **Note:** The first time you run diarization, it may take a few minutes to download the models.
            Subsequent runs will be faster as models are cached.
            """)


if __name__ == "__main__":
    main()

