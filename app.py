import streamlit as st
import torch
from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline
from diffusers import DiffusionPipeline as VideoPipeline
import io
from PIL import Image
import time
import gc

# Page configuration
st.set_page_config(
    page_title="AI Media Generator",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for futuristic UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&display=swap');
    
    * {
        font-family: 'Rajdhani', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(45deg, #00f2ff, #0080ff, #8000ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 30px rgba(0, 242, 255, 0.3);
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
        color: #e0e0e0;
    }
    
    .main {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 20px;
        padding: 20px;
        backdrop-filter: blur(10px);
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(45deg, #00f2ff, #0080ff);
        color: #000;
        border: none;
        padding: 15px 30px;
        font-size: 18px;
        font-weight: 700;
        font-family: 'Orbitron', sans-serif;
        border-radius: 50px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 0 20px rgba(0, 242, 255, 0.5);
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 30px rgba(0, 242, 255, 0.8);
    }
    
    .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSelectbox>div>div>select {
        background: rgba(0, 20, 40, 0.6);
        border: 2px solid #00f2ff;
        color: #00f2ff;
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
    }
    
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
        border-color: #0080ff;
        box-shadow: 0 0 15px rgba(0, 128, 255, 0.5);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #0f0f1e 0%, #1a1a2e 100%);
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(0, 242, 255, 0.1), rgba(0, 128, 255, 0.1));
        border: 2px solid #00f2ff;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 0 20px rgba(0, 242, 255, 0.2);
    }
    
    .stAlert {
        background: rgba(0, 242, 255, 0.1);
        border-left: 4px solid #00f2ff;
        border-radius: 10px;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00f2ff, #0080ff, #8000ff);
    }
    
    /* Glowing effect */
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 20px rgba(0, 242, 255, 0.3); }
        50% { box-shadow: 0 0 40px rgba(0, 242, 255, 0.6); }
    }
    
    .glow {
        animation: glow 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []
if 'generation_history' not in st.session_state:
    st.session_state.generation_history = []
if 'current_pipeline' not in st.session_state:
    st.session_state.current_pipeline = None
if 'current_model' not in st.session_state:
    st.session_state.current_model = None

# Function to clear GPU memory
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# Function to load model with caching
@st.cache_resource
def load_model(model_name, model_type):
    """Load and cache the model"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        with st.spinner(f"🚀 Loading {model_name}..."):
            if model_type == "Standard SD":
                pipe = StableDiffusionPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    use_safetensors=True
                )
            elif model_type == "SDXL":
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    use_safetensors=True
                )
            else:  # Generic
                pipe = DiffusionPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    trust_remote_code=True
                )
            
            pipe = pipe.to(device)
            
            # Enable optimizations
            if device == "cuda":
                pipe.enable_attention_slicing()
                pipe.enable_vae_slicing()
                if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
                    try:
                        pipe.enable_xformers_memory_efficient_attention()
                    except:
                        pass
            
        return pipe
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Main header
st.markdown("<h1 style='text-align: center; font-size: 60px;'>🚀 AI MEDIA GENERATOR</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 20px; color: #00f2ff;'>Generate Stunning Images & Videos with HuggingFace Models</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<h2>⚙️ CONFIGURATION</h2>", unsafe_allow_html=True)
    
    # Model selection
    model_type = st.selectbox(
        "Model Type",
        ["Standard SD", "SDXL", "Video (Experimental)", "Custom"]
    )
    
    # Predefined models
    predefined_models = {
        "Standard SD": [
            "runwayml/stable-diffusion-v1-5",
            "stabilityai/stable-diffusion-2-1",
            "prompthero/openjourney-v4",
            "dreamlike-art/dreamlike-photoreal-2.0"
        ],
        "SDXL": [
            "stabilityai/stable-diffusion-xl-base-1.0",
            "stabilityai/sdxl-turbo"
        ],
        "Video (Experimental)": [
            "damo-vilab/text-to-video-ms-1.7b",
            "cerspense/zeroscope_v2_576w"
        ],
        "Custom": []
    }
    
    if model_type != "Custom":
        model_name = st.selectbox(
            "Select Model",
            predefined_models[model_type]
        )
    else:
        model_name = st.text_input(
            "Enter HuggingFace Model Name",
            placeholder="e.g., username/model-name"
        )
    
    st.markdown("---")
    
    # Generation parameters
    st.markdown("<h3>🎛️ Parameters</h3>", unsafe_allow_html=True)
    
    if model_type != "Video (Experimental)":
        num_images = st.slider("Number of Images", 1, 4, 1)
        width = st.select_slider("Width", options=[512, 768, 1024], value=512)
        height = st.select_slider("Height", options=[512, 768, 1024], value=512)
    else:
        num_frames = st.slider("Number of Frames", 8, 32, 16)
        width = st.select_slider("Width", options=[256, 512, 576], value=256)
        height = st.select_slider("Height", options=[256, 512, 320], value=256)
    
    num_inference_steps = st.slider("Inference Steps", 10, 100, 50)
    guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5, 0.5)
    
    seed = st.number_input("Seed (-1 for random)", -1, 999999, -1)
    
    st.markdown("---")
    
    # System info
    st.markdown("<h3>💻 System Info</h3>", unsafe_allow_html=True)
    device = "🟢 CUDA Available" if torch.cuda.is_available() else "🟡 CPU Mode"
    st.info(device)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        st.text(f"GPU: {gpu_name}")
    
    if st.button("🗑️ Clear GPU Memory"):
        clear_gpu_memory()
        st.success("Memory cleared!")

# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("<h2>📝 Prompt Input</h2>", unsafe_allow_html=True)
    
    prompt = st.text_area(
        "Enter your creative prompt",
        height=150,
        placeholder="A futuristic cityscape at night with neon lights, cyberpunk style, highly detailed, 8k...",
        label_visibility="collapsed"
    )
    
    negative_prompt = st.text_area(
        "Negative Prompt (Optional)",
        height=100,
        placeholder="low quality, blurry, distorted...",
        label_visibility="collapsed"
    )
    
    generate_col1, generate_col2 = st.columns([3, 1])
    
    with generate_col1:
        generate_button = st.button("🎨 GENERATE", use_container_width=True)
    
    with generate_col2:
        if st.button("🔄 Clear"):
            st.session_state.generated_images = []
            st.rerun()

with col2:
    st.markdown("<h2>📊 Generation Stats</h2>", unsafe_allow_html=True)
    
    stats_container = st.container()
    with stats_container:
        if st.session_state.generation_history:
            last_gen = st.session_state.generation_history[-1]
            st.markdown(f"""
            <div class='metric-card'>
                <h4>Last Generation</h4>
                <p>⏱️ Time: {last_gen['time']:.2f}s</p>
                <p>🎯 Model: {last_gen['model']}</p>
                <p>📐 Size: {last_gen['size']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='metric-card'>
            <h4>Session Stats</h4>
            <p>🖼️ Total Generated: {len(st.session_state.generation_history)}</p>
            <p>🎨 Current Model: {model_name if model_name else 'None'}</p>
        </div>
        """, unsafe_allow_html=True)

# Generation logic
if generate_button:
    if not prompt:
        st.error("⚠️ Please enter a prompt!")
    elif not model_name:
        st.error("⚠️ Please select or enter a model name!")
    else:
        try:
            # Load model
            pipe = load_model(model_name, model_type)
            
            if pipe is None:
                st.error("Failed to load model. Please check the model name and try again.")
            else:
                # Generate
                start_time = time.time()
                
                # Set seed
                generator = None
                if seed != -1:
                    generator = torch.Generator(device=pipe.device).manual_seed(seed)
                
                with st.spinner("✨ Generating your creation..."):
                    progress_bar = st.progress(0)
                    
                    if model_type != "Video (Experimental)":
                        # Image generation
                        kwargs = {
                            "prompt": prompt,
                            "negative_prompt": negative_prompt if negative_prompt else None,
                            "num_images_per_prompt": num_images,
                            "num_inference_steps": num_inference_steps,
                            "guidance_scale": guidance_scale,
                            "generator": generator,
                            "width": width,
                            "height": height
                        }
                        
                        result = pipe(**kwargs)
                        images = result.images
                        
                        progress_bar.progress(100)
                        
                        # Store images
                        st.session_state.generated_images = images
                        
                        # Record stats
                        gen_time = time.time() - start_time
                        st.session_state.generation_history.append({
                            'time': gen_time,
                            'model': model_name,
                            'size': f"{width}x{height}",
                            'type': 'image'
                        })
                        
                        # Display success
                        st.success(f"✅ Generated {len(images)} image(s) in {gen_time:.2f} seconds!")
                        
                    else:
                        # Video generation
                        st.warning("Video generation is experimental and may take longer...")
                        kwargs = {
                            "prompt": prompt,
                            "num_inference_steps": num_inference_steps,
                            "guidance_scale": guidance_scale,
                            "num_frames": num_frames,
                            "height": height,
                            "width": width
                        }
                        
                        result = pipe(**kwargs)
                        frames = result.frames[0]
                        
                        progress_bar.progress(100)
                        
                        # Store frames as images
                        st.session_state.generated_images = frames
                        
                        # Record stats
                        gen_time = time.time() - start_time
                        st.session_state.generation_history.append({
                            'time': gen_time,
                            'model': model_name,
                            'size': f"{width}x{height}x{num_frames}",
                            'type': 'video'
                        })
                        
                        st.success(f"✅ Generated video with {len(frames)} frames in {gen_time:.2f} seconds!")
                
        except Exception as e:
            st.error(f"❌ Generation failed: {str(e)}")
            st.error("Please check your model name and parameters.")

# Display generated content
if st.session_state.generated_images:
    st.markdown("---")
    st.markdown("<h2>🎨 Generated Content</h2>", unsafe_allow_html=True)
    
    if len(st.session_state.generated_images) > 1:
        cols = st.columns(min(len(st.session_state.generated_images), 3))
        for idx, img in enumerate(st.session_state.generated_images):
            with cols[idx % 3]:
                st.image(img, use_container_width=True, caption=f"Output {idx+1}")
                
                # Download button
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                st.download_button(
                    label=f"⬇️ Download {idx+1}",
                    data=buf.getvalue(),
                    file_name=f"generated_{idx+1}.png",
                    mime="image/png",
                    use_container_width=True
                )
    else:
        st.image(st.session_state.generated_images[0], use_container_width=True)
        
        # Download button
        buf = io.BytesIO()
        st.session_state.generated_images[0].save(buf, format='PNG')
        st.download_button(
            label="⬇️ Download Image",
            data=buf.getvalue(),
            file_name="generated_image.png",
            mime="image/png"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <p style='color: #00f2ff; font-size: 18px;'>Powered by HuggingFace 🤗 Diffusers | Streamlit</p>
    <p style='color: #666; font-size: 14px;'>Generate responsibly. Respect copyright and model licenses.</p>
</div>
""", unsafe_allow_html=True)
