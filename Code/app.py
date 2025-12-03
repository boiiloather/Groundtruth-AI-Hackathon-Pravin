from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import os
import logging
from typing import List, Optional
import requests
import base64
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Creative Studio")

# CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Configuration - supports multiple free providers
# Priority: 1. Hugging Face (free), 2. Groq (free), 3. OpenAI (if API key provided)
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_IMAGE_PROVIDER = os.getenv("IMAGE_PROVIDER", "auto").lower()

# Initialize OpenAI client only if key is provided
openai_client = None
if OPENAI_API_KEY:
    import openai
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

class AdRequest(BaseModel):
    brand_name: str = Field(..., min_length=1, description="Brand name")
    product_description: str = Field(..., min_length=1, description="Product description")
    target_audience: str = Field(..., min_length=1, description="Target audience")
    tone: str = Field(..., min_length=1, description="Tone/style")
    image_provider: Optional[str] = Field(
        default=None,
        description="Preferred image provider: auto, pollinations, huggingface, openai, placeholder",
    )

class AdCreative(BaseModel):
    image_url: str
    caption: str

class AdResponse(BaseModel):
    creatives: List[AdCreative]
    success: bool
    message: Optional[str] = None
    image_provider: Optional[str] = None

def generate_image_prompt(brand_name: str, product_description: str, target_audience: str, tone: str) -> str:
    """Construct a prompt for image generation."""
    return f"""Create a professional, high-quality advertisement image for:
Brand: {brand_name}
Product: {product_description}
Target Audience: {target_audience}
Tone/Style: {tone}

The image should be visually appealing, modern, and suitable for social media advertising. 
Focus on showcasing the product in an attractive way that resonates with the target audience."""

def generate_caption_prompt(brand_name: str, product_description: str, target_audience: str, tone: str) -> str:
    """Construct a prompt for caption generation."""
    return f"""Generate 3 short, engaging ad captions for the following product:

Brand: {brand_name}
Product: {product_description}
Target Audience: {target_audience}
Tone: {tone}

Requirements:
- Each caption should be 5-15 words
- Match the specified tone
- Be catchy and memorable
- Include a call-to-action when appropriate
- Return only the captions, one per line, numbered 1-3"""

async def generate_images_huggingface(prompt: str, num_images: int = 2) -> List[str]:
    """Generate images using Hugging Face Inference API (FREE)."""
    if not HUGGINGFACE_API_KEY:
        raise Exception("Hugging Face API key not configured")

    import asyncio

    image_urls: List[str] = []
    models_to_try = [
        "stabilityai/sdxl-turbo",
        "stabilityai/stable-diffusion-xl-base-1.0",
        "stabilityai/stable-diffusion-2-1",
        "runwayml/stable-diffusion-v1-5",
        "CompVis/stable-diffusion-v1-4",
    ]

    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
        "Accept": "image/png",
        "Content-Type": "application/json",
    }

    for model in models_to_try:
        if len(image_urls) >= num_images:
            break

        api_url = f"https://router.huggingface.co/hf-inference/models/{model}"
        logger.info(f"Attempting Hugging Face model: {model}")
        remaining = num_images - len(image_urls)

        for image_index in range(remaining):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    varied_prompt = prompt if (len(image_urls) == 0 and image_index == 0) else f"{prompt}, variation {image_index + 1}"
                    payload = {
                        "inputs": varied_prompt,
                        "options": {"wait_for_model": True}
                    }

                    response = requests.post(
                        api_url,
                        headers=headers,
                        json=payload,
                        timeout=90
                    )

                    content_type = response.headers.get("content-type", "")
                    logger.info(f"Hugging Face status {response.status_code} for {model}, content-type: {content_type}")

                    if response.status_code == 200 and "image" in content_type:
                        image_data = response.content
                        if len(image_data) < 1000:
                            raise Exception("Received tiny image data")
                        image_base64 = base64.b64encode(image_data).decode("utf-8")
                        image_urls.append(f"data:{content_type};base64,{image_base64}")
                        logger.info(f"Generated image via Hugging Face model {model}")
                        break

                    if response.status_code == 503:
                        wait_time = 10 * (attempt + 1)
                        logger.warning(f"Model {model} loading, retrying in {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue

                    if response.status_code == 429:
                        wait_time = 20 * (attempt + 1)
                        logger.warning(f"Rate limit for {model}, retrying in {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue

                    error_preview = response.text[:200] if response.text else "Unknown error"
                    raise Exception(f"HF error {response.status_code}: {error_preview}")

                except requests.exceptions.Timeout:
                    logger.warning(f"Hugging Face timeout (attempt {attempt + 1})")
                    await asyncio.sleep(10)
                except Exception as e:
                    logger.error(f"Hugging Face model {model} attempt {attempt + 1} failed: {str(e)}")
                    if attempt == max_retries - 1:
                        logger.warning(f"Giving up on current image for model {model}")
                    else:
                        await asyncio.sleep(5)
                else:
                    # success
                    break
            else:
                # if loop not broken, move to next model
                logger.warning(f"Failed to generate image with model {model}, trying next model")
                break

    if not image_urls:
        raise Exception("Hugging Face could not generate images.")

    return image_urls

async def generate_images_openai(prompt: str, num_images: int = 2) -> List[str]:
    """Generate images using OpenAI DALL-E API (requires API key)."""
    if not openai_client:
        raise Exception("OpenAI API key not configured")
    
    image_urls = []
    for i in range(num_images):
        logger.info(f"Generating image {i+1}/{num_images} via OpenAI")
        varied_prompt = prompt if i == 0 else f"{prompt} (variation {i+1})"
        
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=varied_prompt,
            size="1024x1024",
            quality="standard"
        )
        image_url = response.data[0].url
        image_urls.append(image_url)
        logger.info(f"Successfully generated image {i+1} via OpenAI")
    
    return image_urls

async def generate_placeholder_images(prompt: str, num_images: int = 2) -> List[str]:
    """Generate placeholder images using a free placeholder service."""
    # Use placeholder.com or similar service as last resort
    # This ensures we always return something for demo purposes
    placeholder_urls = []
    for i in range(num_images):
        # Use a simple placeholder service
        # Format: https://via.placeholder.com/1024x1024?text=...
        text = prompt[:50].replace(" ", "+").replace("\n", "+")
        placeholder_url = f"https://via.placeholder.com/1024x1024/667eea/ffffff?text=Ad+Creative+{i+1}"
        placeholder_urls.append(placeholder_url)
    logger.info(f"Generated {len(placeholder_urls)} placeholder images")
    return placeholder_urls

def sanitize_prompt_for_url(prompt: str) -> str:
    """Prepare prompt text for use in a URL."""
    return requests.utils.quote(prompt.replace("\n", " ").strip())

async def generate_images_pollinations(prompt: str, num_images: int = 2) -> List[str]:
    """Generate images using Pollinations AI (free, no API key required)."""
    image_urls = []
    import asyncio

    base_url = "https://image.pollinations.ai/prompt"
    sanitized_prompt = sanitize_prompt_for_url(
        f"High-quality advertisement photo, {prompt}"
    )

    for i in range(num_images):
        try:
            logger.info(f"Generating image {i+1}/{num_images} via Pollinations")
            # Add variation seed for diversity
            pollinations_url = f"{base_url}/{sanitized_prompt}?width=1024&height=1024&seed={i*37}"
            response = requests.get(pollinations_url, timeout=60)

            if response.status_code == 200 and len(response.content) > 1000:
                image_base64 = base64.b64encode(response.content).decode("utf-8")
                image_url = f"data:image/png;base64,{image_base64}"
                image_urls.append(image_url)
                logger.info(f"Successfully generated image {i+1} via Pollinations")
            else:
                logger.warning(
                    f"Pollinations returned status {response.status_code}: {response.text[:100]}"
                )
                raise Exception("Pollinations response invalid")
        except Exception as e:
            logger.error(f"Pollinations generation failed: {str(e)}")
            if i == 0:
                raise
            break
        await asyncio.sleep(1)  # brief pause between calls

    if not image_urls:
        raise Exception("Pollinations failed to generate images")

    return image_urls

async def generate_images(prompt: str, num_images: int = 2, provider: str = "auto") -> (List[str], str):
    """Generate images using selected providers. Returns (urls, provider_used)."""
    provider = (provider or "auto").lower()
    
    provider_order: List[str] = []
    
    def add_provider(name: str):
        if name not in provider_order:
            provider_order.append(name)
    
    if provider == "pollinations":
        add_provider("pollinations")
    elif provider == "huggingface":
        add_provider("huggingface")
    elif provider == "openai":
        add_provider("openai")
    elif provider == "placeholder":
        add_provider("placeholder")
    else:
        # auto/default order
        add_provider("pollinations")
        add_provider("huggingface")
        add_provider("openai")
        add_provider("placeholder")
    
    for source in provider_order:
        try:
            if source == "pollinations":
                urls = await generate_images_pollinations(prompt, num_images)
                return urls, "Pollinations"
            if source == "huggingface":
                urls = await generate_images_huggingface(prompt, num_images)
                return urls, "Hugging Face"
            if source == "openai" and openai_client:
                logger.info("Using OpenAI for image generation...")
                urls = await generate_images_openai(prompt, num_images)
                return urls, "OpenAI DALL·E"
            if source == "placeholder":
                logger.info("Using placeholder images")
                urls = await generate_placeholder_images(prompt, num_images)
                return urls, "Placeholder"
        except Exception as e:
            logger.warning(f"{source.capitalize()} provider failed: {str(e)}")
            continue
    
    raise Exception("All image providers failed")

def parse_captions(caption_text: str) -> List[str]:
    """Parse captions from AI response."""
    captions = []
    for line in caption_text.split('\n'):
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith('-')):
            # Remove numbering/bullets
            caption = line.split('.', 1)[-1].strip() if '.' in line else line.lstrip('- ').strip()
            if caption:
                captions.append(caption)
    
    # If parsing failed, split by newlines
    if not captions:
        captions = [c.strip() for c in caption_text.split('\n') if c.strip()]
    
    # Ensure we have at least 2-3 captions
    if len(captions) < 2:
        # Split by periods or create variations
        if len(captions) == 1:
            parts = captions[0].split('.')
            captions = [p.strip() for p in parts if p.strip()][:3]
    
    return captions[:3]  # Return max 3

async def generate_captions_groq(prompt: str) -> List[str]:
    """Generate captions using Groq API (FREE tier, very fast)."""
    if not GROQ_API_KEY:
        raise Exception("Groq API key not configured")
    
    try:
        logger.info("Generating captions with Groq")
        api_url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.1-8b-instant",  # Fast and free
            "messages": [
                {"role": "system", "content": "You are a creative copywriter specializing in short, punchy ad captions."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.8,
            "max_tokens": 200
        }
        
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            caption_text = result["choices"][0]["message"]["content"].strip()
            captions = parse_captions(caption_text)
            logger.info(f"Generated {len(captions)} captions via Groq")
            return captions
        else:
            raise Exception(f"Groq API error: {response.status_code} - {response.text}")
    
    except Exception as e:
        logger.error(f"Error generating captions with Groq: {str(e)}")
        raise

async def generate_captions_huggingface(prompt: str) -> List[str]:
    """Generate captions using Hugging Face Inference API (FREE)."""
    import asyncio
    
    try:
        logger.info("Generating captions with Hugging Face")
        # Using a simpler, faster model
        model = "gpt2"  # Smaller, faster model that's always available
        api_url = f"https://router.huggingface.co/models/{model}"
        
        headers = {"Content-Type": "application/json"}
        if HUGGINGFACE_API_KEY:
            headers["Authorization"] = f"Bearer {HUGGINGFACE_API_KEY}"
        
        # Simplified prompt for GPT-2
        simple_prompt = f"Ad caption for {prompt.split('Brand:')[1].split('Product:')[0].strip() if 'Brand:' in prompt else 'product'}:"
        
        payload = {
            "inputs": simple_prompt,
            "parameters": {
                "max_new_tokens": 100,
                "temperature": 0.9,
                "return_full_text": False,
                "num_return_sequences": 3
            }
        }
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = requests.post(api_url, headers=headers, json=payload, timeout=60)
                
                if response.status_code == 200:
                    result = response.json()
                    # Handle different response formats
                    if isinstance(result, list):
                        if len(result) > 0 and isinstance(result[0], dict):
                            caption_text = result[0].get("generated_text", "")
                        else:
                            caption_text = " ".join([str(r) for r in result[:3]])
                    elif isinstance(result, dict):
                        caption_text = result.get("generated_text", "")
                    else:
                        caption_text = str(result)
                    
                    # Parse and clean up captions
                    captions = parse_captions(caption_text)
                    if not captions:
                        # Fallback: create simple captions from prompt
                        return generate_fallback_captions(prompt)
                    
                    logger.info(f"Generated {len(captions)} captions via Hugging Face")
                    return captions
                elif response.status_code == 503:
                    wait_time = 20 * (attempt + 1)
                    logger.warning(f"Model is loading, waiting {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    continue
                elif response.status_code == 429:
                    logger.warning("Rate limit hit, waiting 30 seconds...")
                    await asyncio.sleep(30)
                    continue
                else:
                    if attempt == max_retries - 1:
                        raise Exception(f"Hugging Face API error: {response.status_code}")
                    await asyncio.sleep(5)
            
            except requests.exceptions.Timeout:
                if attempt == max_retries - 1:
                    raise Exception("Request timeout")
                await asyncio.sleep(5)
        
        # If we get here, all retries failed
        raise Exception("Failed after multiple retries")
    
    except Exception as e:
        logger.error(f"Error generating captions with Hugging Face: {str(e)}")
        raise

async def generate_captions_openai(prompt: str) -> List[str]:
    """Generate captions using OpenAI GPT API (requires API key)."""
    if not openai_client:
        raise Exception("OpenAI API key not configured")
    
    try:
        logger.info("Generating captions with OpenAI")
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a creative copywriter specializing in short, punchy ad captions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=200
        )
        
        caption_text = response.choices[0].message.content.strip()
        captions = parse_captions(caption_text)
        logger.info(f"Generated {len(captions)} captions via OpenAI")
        return captions
    
    except Exception as e:
        logger.error(f"Error generating captions with OpenAI: {str(e)}")
        raise

async def generate_captions(prompt: str) -> List[str]:
    """Generate captions using available API (tries free options first)."""
    # Try Groq first (free, very fast)
    if GROQ_API_KEY:
        try:
            return await generate_captions_groq(prompt)
        except Exception as e:
            logger.warning(f"Groq failed: {str(e)}")
    
    # Try Hugging Face (free)
    try:
        return await generate_captions_huggingface(prompt)
    except Exception as e:
        logger.warning(f"Hugging Face failed: {str(e)}")
        # Fallback to OpenAI if available
        if openai_client:
            try:
                logger.info("Falling back to OpenAI...")
                return await generate_captions_openai(prompt)
            except Exception as e2:
                logger.error(f"OpenAI also failed: {str(e2)}")
                # Last resort: return simple fallback captions
                logger.warning("All APIs failed, using fallback captions")
                return generate_fallback_captions(prompt)
        else:
            # Return fallback captions if no APIs available
            logger.warning("No API keys available, using fallback captions")
            return generate_fallback_captions(prompt)

def generate_fallback_captions(prompt: str) -> List[str]:
    """Generate simple fallback captions when all APIs fail."""
    # Extract key info from prompt
    lines = prompt.split('\n')
    brand = ""
    product = ""
    for line in lines:
        if 'Brand:' in line:
            brand = line.split('Brand:')[1].strip()
        if 'Product:' in line:
            product = line.split('Product:')[1].strip()
    
    # Generate simple captions
    captions = [
        f"Discover {product} from {brand}.",
        f"Experience {product} today.",
        f"{brand}: Quality you can trust."
    ]
    
    if not brand or not product:
        captions = [
            "Discover something amazing.",
            "Experience quality today.",
            "Trusted by many."
        ]
    
    logger.info(f"Generated {len(captions)} fallback captions")
    return captions

@app.post("/generate-ads", response_model=AdResponse)
async def generate_ads(request: AdRequest):
    """Main endpoint to generate ad creatives."""
    try:
        logger.info(f"Received request for brand: {request.brand_name}, product: {request.product_description}")
        image_provider = (request.image_provider or DEFAULT_IMAGE_PROVIDER or "auto").lower()
        
        # Construct prompts
        image_prompt = generate_image_prompt(
            request.brand_name,
            request.product_description,
            request.target_audience,
            request.tone
        )
        caption_prompt = generate_caption_prompt(
            request.brand_name,
            request.product_description,
            request.target_audience,
            request.tone
        )
        
        logger.info(f"Image prompt: {image_prompt[:100]}...")
        logger.info(f"Caption prompt: {caption_prompt[:100]}...")
        
        # Generate images and captions
        images_success = False
        captions_success = False
        image_urls = []
        captions = []
        
        # Try to generate images (2 for faster demo, can be adjusted)
        try:
            image_urls, image_provider_used = await generate_images(
                image_prompt, num_images=2, provider=image_provider
            )
            images_success = True
            logger.info(f"Successfully generated {len(image_urls)} images")
        except Exception as e:
            logger.error(f"Image generation failed: {str(e)}")
            image_urls = []
        
        # Try to generate captions
        try:
            captions = await generate_captions(caption_prompt)
            captions_success = True
            logger.info(f"Successfully generated {len(captions)} captions")
        except Exception as e:
            logger.error(f"Caption generation failed: {str(e)}")
            captions = []
        
        # Combine images and captions
        creatives = []
        if images_success and captions_success:
            # Pair them up
            num_pairs = min(len(image_urls), len(captions))
            for i in range(num_pairs):
                creatives.append(AdCreative(
                    image_url=image_urls[i],
                    caption=captions[i]
                ))
        elif images_success:
            # Only images
            for img_url in image_urls:
                creatives.append(AdCreative(
                    image_url=img_url,
                    caption="Caption generation unavailable"
                ))
        elif captions_success:
            # Only captions (no images)
            for caption in captions:
                creatives.append(AdCreative(
                    image_url="",
                    caption=caption
                ))
        
        if not creatives:
            return AdResponse(
                creatives=[],
                success=False,
                message="Failed to generate both images and captions. Please try again."
            )
        
        message = None
        if not images_success:
            message = "Note: Images could not be generated, but captions are available."
        elif not captions_success:
            message = "Note: Captions could not be generated, but images are available."
        
        logger.info(f"Returning {len(creatives)} creatives")
        return AdResponse(
            creatives=creatives,
            success=True,
            message=message,
            image_provider=image_provider_used if images_success else None,
        )
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred. Please try again.")

@app.get("/")
async def read_root():
    """Serve the main HTML page."""
    return FileResponse("index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/api-status")
async def api_status():
    """Check API key configuration status."""
    status = {
        "huggingface_api_key": "Set" if HUGGINGFACE_API_KEY else "Not Set",
        "groq_api_key": "Set" if GROQ_API_KEY else "Not Set",
        "openai_api_key": "Set" if OPENAI_API_KEY else "Not Set",
        "recommendations": []
    }
    
    if not HUGGINGFACE_API_KEY:
        status["recommendations"].append("Set HUGGINGFACE_API_KEY for better image generation reliability")
    if not GROQ_API_KEY:
        status["recommendations"].append("Set GROQ_API_KEY for faster caption generation (already working)")
    
    return status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

