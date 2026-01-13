"""
Example usage of Speaker Identifier with both Resemblyzer and Pyannote models
"""

from speaker_identifier import SpeakerIdentifier
import os

def test_resemblyzer():
    """Test speaker identification with Resemblyzer model"""
    print("=" * 60)
    print("Testing with RESEMBLYZER model")
    print("=" * 60)
    
    # Initialize with Resemblyzer
    identifier = SpeakerIdentifier(
        voice_samples_dir="voice_samples",
        model_type="resemblyzer"
    )
    
    # Load reference voices
    print("\nLoading reference voices...")
    identifier.load_reference_voices()
    print(f"Loaded {len(identifier.speaker_names)} speakers: {identifier.speaker_names}")
    
    # Analyze a test audio file
    test_audio = "path/to/your/test_audio.mp3"  # Replace with actual path
    if os.path.exists(test_audio):
        print(f"\nAnalyzing audio: {test_audio}")
        results = identifier.get_overall_speaker_matches(test_audio)
        
        print("\nTop 3 matches:")
        for i, match in enumerate(results[:3], 1):
            print(f"{i}. {match['speaker']}: {match['average_percentage']:.2f}% "
                  f"(max: {match['max_percentage']:.2f}%)")
    else:
        print(f"\nTest audio not found: {test_audio}")
        print("Please update the 'test_audio' variable with a valid audio file path")


def test_pyannote():
    """Test speaker identification with Pyannote model"""
    print("\n" + "=" * 60)
    print("Testing with PYANNOTE model")
    print("=" * 60)
    
    # Check if HF_TOKEN is set
    if not os.environ.get("HF_TOKEN"):
        print("\n⚠️  Warning: HF_TOKEN environment variable not set")
        print("To use Pyannote, please:")
        print("1. Get your token from https://huggingface.co/settings/tokens")
        print("2. Accept agreement at https://huggingface.co/pyannote/embedding")
        print("3. Set environment variable: export HF_TOKEN='your_token'")
        return
    
    try:
        # Initialize with Pyannote
        identifier = SpeakerIdentifier(
            voice_samples_dir="voice_samples",
            model_type="pyannote"
        )
        
        # Load reference voices
        print("\nLoading reference voices with Pyannote...")
        identifier.load_reference_voices()
        print(f"Loaded {len(identifier.speaker_names)} speakers: {identifier.speaker_names}")
        
        # Analyze a test audio file
        test_audio = "path/to/your/test_audio.mp3"  # Replace with actual path
        if os.path.exists(test_audio):
            print(f"\nAnalyzing audio: {test_audio}")
            results = identifier.get_overall_speaker_matches(test_audio)
            
            print("\nTop 3 matches:")
            for i, match in enumerate(results[:3], 1):
                print(f"{i}. {match['speaker']}: {match['average_percentage']:.2f}% "
                      f"(max: {match['max_percentage']:.2f}%)")
        else:
            print(f"\nTest audio not found: {test_audio}")
            print("Please update the 'test_audio' variable with a valid audio file path")
            
    except Exception as e:
        print(f"\n❌ Error with Pyannote: {e}")
        print("\nPlease ensure:")
        print("1. pyannote.audio is installed: pip install pyannote.audio")
        print("2. You've accepted the user agreement")
        print("3. HF_TOKEN is set correctly")


def compare_models():
    """Compare both models on the same audio"""
    print("\n" + "=" * 60)
    print("Comparing RESEMBLYZER vs PYANNOTE")
    print("=" * 60)
    
    test_audio = "path/to/your/test_audio.mp3"  # Replace with actual path
    
    if not os.path.exists(test_audio):
        print(f"\nTest audio not found: {test_audio}")
        print("Please update the 'test_audio' variable with a valid audio file path")
        return
    
    # Test with Resemblyzer
    print("\n1. Resemblyzer Results:")
    resemblyzer_id = SpeakerIdentifier("voice_samples", model_type="resemblyzer")
    resemblyzer_id.load_reference_voices()
    resemblyzer_results = resemblyzer_id.get_overall_speaker_matches(test_audio)
    
    for match in resemblyzer_results[:3]:
        print(f"   {match['speaker']}: {match['average_percentage']:.2f}%")
    
    # Test with Pyannote (if available)
    if os.environ.get("HF_TOKEN"):
        try:
            print("\n2. Pyannote Results:")
            pyannote_id = SpeakerIdentifier("voice_samples", model_type="pyannote")
            pyannote_id.load_reference_voices()
            pyannote_results = pyannote_id.get_overall_speaker_matches(test_audio)
            
            for match in pyannote_results[:3]:
                print(f"   {match['speaker']}: {match['average_percentage']:.2f}%")
        except Exception as e:
            print(f"   ⚠️  Pyannote failed: {e}")
    else:
        print("\n2. Pyannote: Skipped (HF_TOKEN not set)")


if __name__ == "__main__":
    print("Speaker Identifier Example Usage\n")
    
    # Test Resemblyzer
    test_resemblyzer()
    
    # Test Pyannote
    test_pyannote()
    
    # Compare both models (uncomment to use)
    # compare_models()
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

