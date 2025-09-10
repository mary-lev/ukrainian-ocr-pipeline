# 🇺🇦 Ukrainian OCR Pipeline - Google Colab Setup

## 🚀 Quick Start in Google Colab

The Ukrainian OCR Pipeline is designed to run efficiently in Google Colab with GPU acceleration for fast processing of historical Ukrainian documents.

## 📋 Prerequisites

1. **Google Account** - Required for Google Colab access
2. **GPU Runtime** - Essential for fast processing
   - Go to `Runtime` → `Change runtime type` → Select `GPU` (T4, V100, or A100)
   - Verify GPU: Run `!nvidia-smi` in a cell

## 🔧 Installation Steps

### Step 1: Clone Repository
```python
# Clone the Ukrainian OCR pipeline repository
!git clone https://github.com/mary-lev/ukrainian-ocr-pipeline.git
%cd ukrainian-ocr-pipeline
```

### Step 2: Install Dependencies
```python
# Install the package from source with Colab dependencies
!pip install -e .[colab] --quiet

# Install additional dependencies for Colab
!pip install ipywidgets --quiet

print("✅ Installation from GitHub complete!")
```

### Step 3: Verify Installation
```python
# Check GPU availability and setup
import ukrainian_ocr
from ukrainian_ocr.utils.gpu import check_gpu_availability, setup_colab_gpu

# Check GPU
gpu_info = setup_colab_gpu()

if gpu_info['cuda_available']:
    print(f"🎉 GPU detected: {gpu_info['gpu_names'][0]}")
    print(f"💾 GPU Memory: {gpu_info['gpu_memory'][0]:.1f}GB")
    print(f"🔥 Recommended device: {gpu_info['recommended_device']}")
else:
    print("⚠️ No GPU detected. Enable GPU: Runtime -> Change runtime type -> GPU")
    print("💻 Will use CPU (slower processing)")
```

## 📤 Upload Your Documents

```python
from google.colab import files
import os

# Create upload directory
os.makedirs('/content/images', exist_ok=True)

# Upload files
print("📤 Select your historical document images to upload:")
uploaded = files.upload()

# Move uploaded files to images directory
for filename in uploaded.keys():
    os.rename(filename, f'/content/images/{filename}')
    print(f"✅ Uploaded: {filename}")

# List uploaded files
image_files = [f'/content/images/{f}' for f in os.listdir('/content/images')]
print(f"\n📊 Total images uploaded: {len(image_files)}")
```

## ⚙️ Basic Configuration

```python
from ukrainian_ocr import UkrainianOCRPipeline, OCRConfig

# Create optimized configuration for Colab
config = OCRConfig()
config.update_for_colab()  # Optimize for Colab environment

# Initialize pipeline
pipeline = UkrainianOCRPipeline(
    config=config,
    device='auto',  # Auto-detect best device
    verbose=True
)

print("✅ Pipeline initialized successfully!")
```

## 🔄 Process Documents

```python
# Process all uploaded images
import time

print(f"🔄 Processing {len(image_files)} image(s)...")

# Create output directory
output_dir = '/content/ocr_results'
os.makedirs(output_dir, exist_ok=True)

# Start processing
start_time = time.time()

if len(image_files) == 1:
    # Single image processing
    results = [pipeline.process_single_image(
        image_files[0], 
        output_dir=output_dir,
        save_intermediate=True
    )]
else:
    # Batch processing
    results = pipeline.process_batch(
        image_files, 
        output_dir=output_dir,
        save_intermediate=True
    )

total_time = time.time() - start_time

# Display results summary
successful = sum(1 for r in results if r['success'])
failed = len(results) - successful

print(f"\n🎉 Processing complete!")
print(f"✅ Successful: {successful}/{len(results)}")
print(f"⏱️ Total time: {total_time:.1f}s")
print(f"📊 Average per image: {total_time/len(results):.1f}s")
```

## 💾 Download Results

```python
import shutil
import zipfile

# Create a zip file with all results
zip_path = '/content/ukrainian_ocr_results.zip'

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arc_path = os.path.relpath(file_path, '/content')
            zipf.write(file_path, arc_path)

zip_size_mb = os.path.getsize(zip_path) / 1024 / 1024
print(f"📦 Created results archive: ukrainian_ocr_results.zip ({zip_size_mb:.1f}MB)")

# Download the zip file
files.download(zip_path)
print("✅ Results downloaded successfully!")
```

## ⚡ Performance Expectations

### **Google Colab Free (T4 GPU)**
- **Processing speed**: 10-15 seconds per document page
- **Memory usage**: ~4-6GB GPU memory
- **Batch size**: 4-8 images simultaneously
- **Session limit**: 12 hours
- **Monthly usage**: ~100-200 GPU hours

### **Google Colab Pro (Better GPUs)**
- **Processing speed**: 5-10 seconds per document page
- **Available GPUs**: V100, A100
- **Longer sessions**: Up to 24 hours
- **Priority access**: Faster startup and fewer disconnections

## 🛠️ Troubleshooting

### **GPU Not Available**
```python
# Check if GPU is enabled
!nvidia-smi

# If command fails:
# 1. Go to Runtime -> Change runtime type
# 2. Select GPU (T4 recommended)
# 3. Click Save and restart
```

### **Memory Issues**
```python
# Clear GPU memory
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    
# Reduce batch size in config
config.batch_size = 2  # Reduce if getting OOM errors
```

### **Installation Issues**
```python
# Force reinstall
!pip uninstall ukrainian-ocr-pipeline -y
!git pull origin main  # Get latest changes
!pip install -e .[colab] --force-reinstall
```

### **Upload Issues**
```python
# Alternative upload method using Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Access files from Drive
image_files = ['/content/drive/MyDrive/documents/image1.jpg']
```

## 📊 Monitoring Progress

```python
# Monitor GPU memory usage
from ukrainian_ocr.utils.gpu import monitor_gpu_memory

memory_stats = monitor_gpu_memory()
for gpu_id, stats in memory_stats.items():
    print(f"📊 {gpu_id.upper()} Memory Usage:")
    print(f"  Allocated: {stats['allocated']:.1f}GB")
    print(f"  Total: {stats['total']:.1f}GB")
    print(f"  Utilization: {stats['utilization']:.1f}%")
```

## 🔧 Advanced Configuration

### **Custom Settings**
```python
config = OCRConfig()

# GPU optimization
config.device = 'cuda'
config.batch_size = 8  # Adjust based on GPU memory

# OCR settings  
config.ocr.preprocessing = False  # Faster processing
config.ocr.num_beams = 1  # Greedy search for speed

# NER settings
config.ner.backend = "spacy"
config.ner.confidence_threshold = 0.7

# Enable person region extraction
config.post_processing.extract_person_regions = True
config.post_processing.clustering_eps = 300
```

### **Batch Processing Optimization**
```python
# For many small images
config.batch_size = 8

# For few large images  
config.batch_size = 2

# Auto-optimize based on GPU memory
config.batch_size = None  # Auto-determine
```

## 🎯 Best Practices

1. **Enable GPU Runtime** - Essential for good performance
2. **Process in batches** - More efficient than single images
3. **Monitor memory** - Adjust batch size if getting OOM errors
4. **Save intermediate steps** - Helpful for debugging
5. **Download results frequently** - Colab sessions can timeout
6. **Use the notebook template** - Pre-configured for best performance

## 🚀 Using the Complete Notebook

The easiest way to get started is using our pre-built Colab notebook:

1. **Open the notebook**: `examples/Ukrainian_OCR_Colab_Demo.ipynb`
2. **Enable GPU**: Runtime → Change runtime type → GPU
3. **Run all cells**: Runtime → Run all
4. **Upload your images** when prompted
5. **Download results** at the end

The notebook includes:
- ✅ Automatic setup and installation
- ✅ GPU optimization and checking  
- ✅ Interactive file upload
- ✅ Progress tracking with visual indicators
- ✅ Automatic results download
- ✅ Error handling and troubleshooting

---

**Need help?** Open an issue at: https://github.com/mary-lev/ukrainian-ocr-pipeline/issues