# 🎨 AI Creative Studio: The Instant Ad Generator

**Tagline:** An AI-powered creative engine that generates professional ad creatives (image + caption) for any brand and product in under 20 seconds, built for hackathon speed and demo reliability.

---

## 1. The Problem (Real World Scenario)

**Context:** During my research into marketing workflows, I identified a critical bottleneck: Marketing teams and small businesses spend hours creating ad creatives. They need to hire designers, write copy, iterate on concepts, and coordinate between multiple tools—all for a single campaign.

**The Pain Point:** This process is expensive, time-consuming, and requires multiple skill sets. A startup launching a new product might need 2-3 ad variations, but creating them manually takes days and costs hundreds of dollars in design fees. For hackathons and rapid prototyping, this delay kills momentum.

**My Solution:** I built AI Creative Studio, a single-page web application. You simply fill in four fields (brand, product, audience, tone), click "Generate Ads," and 20 seconds later, you receive 2-3 professionally designed ad images with matching captions, ready to download and use.

---

## 2. Expected End Result

**For the User:**

- **Input:** Fill out a simple form with brand name, product description, target audience, and desired tone.
- **Action:** Click "Generate Ads" and wait 10-20 seconds.
- **Output:** Receive a grid of 2-3 ad creatives, each containing:
  - A high-quality, professional advertisement image (1024x1024px)
  - A catchy, tone-matched caption (5-15 words)
  - A download button to save the image locally

**Example Workflow:**

1. Brand: "EcoSip"
2. Product: "Reusable stainless-steel water bottle"
3. Audience: "College students"
4. Tone: "Casual, eco-friendly"
5. **Result:** 2-3 lifestyle product images + captions like "Stay hydrated, skip the plastic." or "Your planet, your choice."

---

## 3. Technical Approach

I wanted to build a system that is **Demo-Ready** and **Reliable**, moving beyond simple API calls to a robust, error-resilient pipeline that handles failures gracefully.

**System Architecture:**

**Frontend (Event-Driven UI):**
- A single-page HTML application with vanilla JavaScript
- Real-time form validation with inline error messages
- Loading states and error handling for better UX
- Auto-detection of API endpoint (works whether served from FastAPI or opened directly)

**Backend (FastAPI Microservice):**
- RESTful API endpoint `/generate-ads` that accepts JSON payloads
- CORS enabled for cross-origin requests
- Structured logging for debugging during live demos

**AI Orchestration (Parallel Processing):**

**Image Generation:**
- I use OpenAI's DALL-E 3 API to generate high-quality advertisement images
- Each image is generated with a carefully constructed prompt that includes brand context, product details, target audience, and tone
- **Decision:** I chose sequential generation (2 images) over parallel to respect API rate limits and ensure reliability during demos

**Caption Generation:**
- I pass the same brand context to GPT-3.5-turbo with a specialized system prompt
- The AI is instructed to act as a "creative copywriter" specializing in short, punchy ad captions
- **Guardrail:** I implemented parsing logic to extract multiple captions from the AI response, with fallback handling if the format varies

**Error Resilience:**
- I implemented a "Partial Success" pattern: if image generation fails but captions succeed (or vice versa), the system still displays whatever was successfully generated
- Full error details are logged to the server console for debugging, while users see friendly, actionable error messages
- Input validation ensures non-empty fields before making API calls

**Result Display:**
- Images and captions are paired and displayed in a responsive grid layout
- Native browser download functionality (no server-side storage needed)
- Warning messages appear if only partial results are available

---

## 4. Tech Stack

- **Language:** Python 3.8+
- **Backend Framework:** FastAPI (lightweight, async-capable, auto-documentation)
- **Frontend:** Vanilla HTML/CSS/JavaScript (no build step, instant loading)
- **Image Generation:** 
  - **Primary (Free):** Hugging Face Stable Diffusion XL
  - **Fallback:** OpenAI DALL-E 3 (if API key provided)
- **Text Generation:**
  - **Primary (Free):** Groq API (Llama 3.1, 14,400 free requests/day)
  - **Secondary (Free):** Hugging Face Mistral-7B
  - **Fallback:** OpenAI GPT-3.5-turbo (if API key provided)
- **Data Validation:** Pydantic models
- **Server:** Uvicorn (ASGI server)
- **Deployment:** Single-file backend, static HTML frontend (no database, no complex infrastructure)
- **HTTP Client:** Requests library for API calls

---

## 5. Challenges & Learnings

This project wasn't easy. Here are two major hurdles I overcame:

**Challenge 1: Free API Integration & Fallback Strategy**

**Issue:** Initially, the app only supported OpenAI's paid APIs. For hackathons, developers need free options. I needed to integrate multiple free APIs with intelligent fallback logic.

**Solution:** I implemented a multi-provider architecture that tries free APIs first (Hugging Face, Groq) and falls back to OpenAI only if available. The system gracefully handles API failures and shows partial results. I also added support for base64 image encoding (from Hugging Face) alongside URL-based images (from OpenAI).

**Challenge 2: Error Handling During Live Demos**

**Issue:** During a live demo, if the OpenAI API timed out or returned an error, the entire flow would break, leaving the user with nothing. This is a demo-killer.

**Solution:** I implemented a "Partial Success" architecture. The system tries to generate both images and captions independently. If one fails, it still displays the successful results with a clear warning message. This ensures that even in failure scenarios, the demo can continue and show value.

**Bonus Learning: CORS & Static File Serving**

**Issue:** Initially, I had the frontend calling `localhost:8000` hardcoded, which broke when serving the HTML from FastAPI directly.

**Solution:** I implemented auto-detection of the API URL based on the current page origin, and added a root endpoint in FastAPI to serve the HTML file directly. This makes the app work in multiple deployment scenarios.

---

## 6. Visual Proof

### Input Form (Before)
*[Screenshot: Clean form with brand, product, audience, and tone fields]*

### Generation in Progress
*[Screenshot: Loading spinner with "Generating your ad creatives..." message]*

### Generated Creatives (After)
*[Screenshot: Grid of 2-3 ad images with captions and download buttons]*

### Example Output: EcoSip Campaign
*[Screenshot: Lifestyle water bottle images with eco-friendly captions]*

---

## 7. How to Run

### Option 1: Free APIs (Recommended for Hackathons)

The app now supports **100% FREE APIs** with no credit card required:

```bash
# 1. Clone Repository
git clone <repository-url>
cd Ground-truth-AI-hackathon

# 2. Install Dependencies
pip install -r requirements.txt

# 3. Get Free API Keys (Choose one or both):

# A. Hugging Face (FREE - No credit card needed)
#    - Sign up at https://huggingface.co/join
#    - Go to https://huggingface.co/settings/tokens
#    - Create a token (read access is enough)
#    - Set environment variable:
#      Windows (PowerShell): $env:HUGGINGFACE_API_KEY="your-token-here"
#      Linux/Mac: export HUGGINGFACE_API_KEY="your-token-here"

# B. Groq (FREE - Very fast, 14,400 requests/day)
#    - Sign up at https://console.groq.com/
#    - Get your API key from https://console.groq.com/keys
#    - Set environment variable:
#      Windows (PowerShell): $env:GROQ_API_KEY="your-key-here"
#      Linux/Mac: export GROQ_API_KEY="your-key-here"

# 4. Start Backend Server
python app.py

# 5. Open Frontend
# Navigate to http://localhost:8000 (FastAPI serves the HTML)
```

**Free API Priority:**
1. **Images**: Hugging Face Stable Diffusion (free, no API key needed for public models)
2. **Captions**: Groq (free, 14,400 requests/day) → Falls back to Hugging Face if unavailable

### Option 2: OpenAI (If you have free credits)

OpenAI offers **$18 free trial credits** for new users (valid 90 days):

```bash
# Set OpenAI API key (optional, only if you want to use OpenAI)
# Windows (PowerShell)
$env:OPENAI_API_KEY="your-openai-api-key-here"

# Linux/Mac
export OPENAI_API_KEY="your-openai-api-key-here"
```

The app will automatically use OpenAI as a fallback if free APIs fail.

### Quick Start (No API Keys Needed!)

You can run the app **without any API keys** - Hugging Face public models work without authentication (though slower):

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:8000
```

Note: Without API keys, you may experience slower generation and occasional model loading delays.

**Quick Test:**
1. Fill in the form with:
   - Brand: "EcoSip"
   - Product: "Reusable stainless-steel water bottle"
   - Audience: "College students"
   - Tone: "Casual, eco-friendly"
2. Click "Generate Ads"
3. Wait 10-20 seconds
4. View and download your creatives!

---

## 8. API Documentation

### POST `/generate-ads`

Generate ad creatives from form inputs.

**Request Body:**
```json
{
  "brand_name": "EcoSip",
  "product_description": "Reusable stainless-steel water bottle",
  "target_audience": "College students",
  "tone": "Casual, eco-friendly"
}
```

**Response:**
```json
{
  "success": true,
  "creatives": [
    {
      "image_url": "https://...",
      "caption": "Stay hydrated, skip the plastic."
    },
    {
      "image_url": "https://...",
      "caption": "Your planet, your choice."
    }
  ],
  "message": null
}
```

### GET `/health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

---

## 9. Architecture Decisions

**Why FastAPI?** Fast, modern, async-capable, and auto-generates API documentation. Perfect for rapid prototyping.

**Why Vanilla JavaScript?** No build step, instant loading, works everywhere. For a 3-hour hackathon project, simplicity wins.

**Why No Database?** All data is ephemeral. Images are generated on-demand and downloaded directly. This keeps the system lightweight and demo-friendly.

**Why Sequential Image Generation?** DALL-E 3 API limitations require sequential requests. I optimized for reliability over speed, ensuring the demo works consistently.

---

## 10. Future Enhancements

- **Batch Processing:** Generate creatives for multiple products at once
- **Style Presets:** Pre-defined tone templates (e.g., "Minimalist", "Bold", "Luxury")
- **A/B Testing:** Generate variations with different messaging strategies
- **Export Options:** Download all creatives as a ZIP file
- **History:** Store recent generations (with user consent)

---

**Built for hackathon purposes. Ready for demo. Zero manual steps.**
