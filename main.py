import traceback
import re
# from pydantic import BaseModel
from fastapi import Request
import os
import base64
import mimetypes
import yaml
import numpy as np
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from tenacity import retry, wait_exponential, stop_after_attempt
from openai import OpenAI
# from google import genai
import google.generativeai as genai
from google.generativeai import GenerativeModel

# from genai.schemas import Content, Part

from dotenv import load_dotenv
load_dotenv()


# Initialize clients

openai_client = OpenAI(
    api_key=os.getenv("AIPIPE_API_KEY"),
    base_url=os.getenv("AIPIPE_BASE_URL")
)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# genai.configure(api_key="your-gemini-api-key")
# gemini_model = "models/gemini-2.0-flash"
# genai_client = genai

# Load .npz file
db = np.load("successful_embeddings_with_metadata.npz", allow_pickle=True)
stored_embeddings = db["embeddings"]
stored_metadata = db["metadata"]  # Actual text chunks

# Initialize app
app = FastAPI()

# --------- Schemas ---------


class QuestionItem(BaseModel):
    text: str
    image: Optional[str] = None

    @field_validator("image")
    def validate_base64(cls, val):
        if val:
            try:
                base64.b64decode(val)
            except Exception:
                raise ValueError("Invalid base64 image")
        return val


class QuestionSet(BaseModel):
    questions: List[QuestionItem]

# --------- Embedding Logic ---------


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(5))
def get_embedding(text: str):
    res = openai_client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return np.array(res.data[0].embedding, dtype=np.float32)

# --------- Cosine Similarity ---------


def cosine_similarity(vec, matrix):
    from numpy.linalg import norm
    return matrix @ vec / (norm(matrix, axis=1) * norm(vec))

# --------- Image Description ---------


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def describe_image_gemini_2(base64_img: str) -> str:
    try:
        img_data = base64.b64decode(base64_img)
        mime = mimetypes.guess_type("file.png")[0] or "image/png"
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(
            [
                "Describe the content of this image in detail, focusing on any text, objects, or relevant features that could help answer questions about it.",
                genai.types.content.Part.from_data(
                    data=img_data, mime_type=mime)
            ],
            generation_config={"temperature": 0.3, "max_output_tokens": 512}
        )
        return response.text.strip()
    except Exception as e:
        return "[Image description failed]"
        
# ------new version---------google-generativeai

def call_gemini_llm(question_text: str, img_desc: str, top_chunks: list):
    prompt = f"""Using the following context, answer the question clearly and helpfully.

Question: {question_text}
Image Description: {img_desc}

Relevant Information:
{chr(10).join(top_chunks)}
"""

    model = genai.GenerativeModel("gemini-2.0-flash")

    response = model.generate_content(
        [prompt],
        generation_config={"temperature": 0.7, "max_output_tokens": 1024}
    )

    return response.text.strip()


# ---------Link parser answer formatter  -------------


def format_gemini_response(text: str) -> dict:
    # Extract markdown-style links: [label](url)
    link_matches = re.findall(r"\[([^\]]+)\]\((https?://[^\)]+)\)", text)
    links = [{"text": label, "url": url} for label, url in link_matches]

    # Remove markdown links from the answer text
    clean_text = re.sub(r"\[([^\]]+)\]\((https?://[^\)]+)\)", r"\1", text)

    return {
        "answer": clean_text.strip(),
        "links": links
    }


# --------- API Endpoint ---------


class SimpleQuestion(BaseModel):
    question: str
    image: Optional[str] = None


@app.post("/api/")
async def answer_question(payload: SimpleQuestion):
    question_text = payload.question
    img_desc = ""
    # Step 1: Describe image (if present)
    if payload.image:
        img_desc = describe_image_gemini_2(payload.image)  # Gemini 2.0 Flash
    full_query = question_text + " " + img_desc
    embed = get_embedding(full_query)
    # step 3 chnaged:
    embed = np.array(embed, dtype=np.float32)
    stored_matrix = np.array(stored_embeddings, dtype=np.float32)
    similarities = np.dot(stored_matrix, embed) / (
        np.linalg.norm(stored_matrix, axis=1) * np.linalg.norm(embed)
    )
    top_idx = np.argsort(similarities)[-10:][::-1]
    top_chunks = [stored_metadata[i] for i in top_idx]
    # 3
    # Step 4: Generate answer using Gemini
    prompt = f"""Answer the following question using the provided context.
If helpful, include relevant link URLs from the text as clickable markdown links.

Question: {question_text}
Image Description: {img_desc}

Context:
{chr(10).join(top_chunks)}
"""
    model = GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(
        [prompt],
        generation_config={"temperature": 0.7, "max_output_tokens": 1024}
    )

    parsed = format_gemini_response(response.text.strip())
    return parsed

'''
@app.post("/generate-answer/")
async def process_yaml(file: UploadFile):
    try:
        raw = await file.read()
        parsed_yaml = yaml.safe_load(raw)
        validated = QuestionSet(**parsed_yaml)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid YAML: {str(e)}"})
    results = []
    for q in validated.questions:
        question_text = q.text
        img_desc = ""

        if q.image:
            img_desc = describe_image_gemini_2(q.image)
        combined_text = question_text + " " + img_desc
        embed = get_embedding(combined_text)
        embed = np.array(embed, dtype=np.float32)
        stored_matrix = np.array(stored_embeddings, dtype=np.float32)
        similarities = np.dot(stored_matrix, embed) / (
            np.linalg.norm(stored_matrix, axis=1) * np.linalg.norm(embed)
        )
        top_idx = np.argsort(similarities)[-10:][::-1]
        top_chunks = [stored_metadata[i] for i in top_idx]

    final_response = call_gemini_llm(question_text, img_desc, top_chunks)

    results.append({
        "question": question_text,
        "image_description": img_desc,
        "response": final_response
    })

    return {"responses": results}
'''

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

#handler = Mangum(app)
