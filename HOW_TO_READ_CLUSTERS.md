# How to Read Cluster Information from Test Output

When you run `test_model_output.py`, the output text file contains detailed cluster information. Here's how to understand it:

## Finding Cluster Information

Look for these sections in the output file:

### 1. **CLUSTERING ANALYSIS** Section

This section shows:
- How many segments were created from your audio
- How many embeddings were extracted
- How many clusters were created

### 2. **Cluster Distribution**

Shows how many segments belong to each cluster:
```
Cluster distribution:
  Cluster 0: X segments
  Cluster 1: Y segments
```

**What it means:**
- **Cluster 0** = Speaker 1 (Person 1)
- **Cluster 1** = Speaker 2 (Person 2)
- More segments in a cluster = that speaker talked more

### 3. **Segment-to-Cluster Mapping**

Shows which time segments belong to which cluster:
```
Segment    Start (s)    End (s)    Duration (s)    Cluster    Speaker
Segment 0  0.00         3.00       3.00            Cluster 0   Person 1
Segment 1  3.00         6.00       3.00            Cluster 1   Person 2
Segment 2  6.00         9.00       3.00            Cluster 0   Person 1
```

**What it means:**
- Each row shows a time segment and which speaker (cluster) it belongs to
- You can see when each speaker was talking

### 4. **CLUSTER SUMMARY**

Detailed summary for each cluster:
```
Cluster 0 (Speaker: Person 1):
  Number of segments: 5
  Total duration: 15.00 seconds (50.0% of audio)
  Time ranges:
    0.00s - 3.00s
    6.00s - 9.00s
    ...
```

**What it means:**
- Shows total speaking time for each speaker
- Lists all time ranges when that speaker was talking
- Percentage shows how much of the audio each speaker occupies

### 5. **Cosine Similarity Matrix**

Shows how similar different segments are:
```
      Seg0   Seg1   Seg2   ...
Seg0  1.000  0.850  0.920  ...
Seg1  0.850  1.000  0.750  ...
```

**What it means:**
- Values range from -1 to 1 (usually 0 to 1 for voice similarity)
- **Higher values (0.8-1.0)** = Very similar voices (likely same speaker)
- **Lower values (0.0-0.5)** = Different voices (likely different speakers)
- Segments in the same cluster should have high similarity
- Segments in different clusters should have lower similarity

### 6. **CLUSTER QUALITY METRICS**

Shows how well the clustering separated speakers:
```
Average similarity WITHIN clusters: 0.8500
  (Higher is better - segments in same cluster should be similar)
Average similarity BETWEEN clusters: 0.4500
  (Lower is better - different clusters should be dissimilar)

Cluster separation score: 0.4000
  ✓ Good cluster separation - speakers are well distinguished
```

**What it means:**
- **Separation score > 0.1**: Good - speakers are well separated
- **Separation score 0 to 0.1**: Moderate - speakers may be similar
- **Separation score < 0**: Poor - clusters may not represent different speakers

## Quick Interpretation Guide

### If you see 2 clusters:
✅ **Good!** The system detected 2 different speakers
- Check the "Cluster Summary" to see speaking time for each
- Check "Segment-to-Cluster Mapping" to see when each speaker talked

### If you see only 1 cluster:
❌ **Problem!** Only one speaker detected
- Check "Model predicted speaker IDs" - if only one ID, the model isn't detecting multiple speakers
- Check "Cluster Quality Metrics" - if separation is poor, clustering may have failed

### If similarity within clusters is low (<0.7):
⚠️ **Warning!** Segments in the same cluster aren't very similar
- May indicate poor clustering
- Speakers might be too similar for the model to distinguish

### If similarity between clusters is high (>0.7):
⚠️ **Warning!** Different clusters are too similar
- May indicate the same speaker was split into multiple clusters
- Or speakers have very similar voices

## Example Output Interpretation

```
Cluster distribution:
  Cluster 0: 3 segments
  Cluster 1: 2 segments

Cluster 0 (Speaker: Person 1):
  Total duration: 9.00 seconds (60.0% of audio)
  Time ranges:
    0.00s - 3.00s
    6.00s - 9.00s
    12.00s - 15.00s

Cluster 1 (Speaker: Person 2):
  Total duration: 6.00 seconds (40.0% of audio)
  Time ranges:
    3.00s - 6.00s
    9.00s - 12.00s
```

**Interpretation:**
- Person 1 spoke for 9 seconds (60% of audio) in 3 segments
- Person 2 spoke for 6 seconds (40% of audio) in 2 segments
- They alternated speaking (Person 1 → Person 2 → Person 1 → Person 2 → Person 1)

## Tips

1. **Check the file**: Open the `.txt` output file in any text editor
2. **Search for "CLUSTER"**: Use Ctrl+F to find cluster-related sections quickly
3. **Look at the summary**: The "CLUSTER SUMMARY" section gives the easiest overview
4. **Check quality metrics**: If separation is poor, the clustering may need adjustment

