import os
import json
import uuid
import base64
import time
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel

# Structured logging
from logging_config import setup_logging
setup_logging()

# API Gateway middleware
from middleware import GatewayMiddleware, get_metrics, record_llm_call, record_rag_retrieval

# Model factory
from models import get_model_config_for_display

from agent import medical_agent_with_kg, generate_treatment_table_pdf

app = FastAPI(title="医疗智能体 - Medical AI Agent v2.0")
app.add_middleware(GatewayMiddleware)

# In-memory store
conversations_store = {}
CONVERSATIONS_DIR = "data/conversations"
os.makedirs(CONVERSATIONS_DIR, exist_ok=True)
os.makedirs("data/uploads", exist_ok=True)


# ========== Data Models ==========

class PatientQuery(BaseModel):
    patient_id: str
    patient_name: str
    question: str
    current_medications: list = []
    conversation_id: str | None = None


class LoginRequest(BaseModel):
    username: str
    password: str = ""


class OCRRequest(BaseModel):
    image_path: str


# ========== System Info ==========

@app.get("/api/health")
def health():
    return {"status": "ok", "model": get_model_config_for_display()}


@app.get("/metrics")
def metrics():
    return PlainTextResponse(get_metrics(), media_type="text/plain")


# ========== Auth ==========

@app.post("/api/login")
def login(data: LoginRequest):
    user_id = data.username.strip()
    if not user_id:
        raise HTTPException(400, "Username required")
    return {"user_id": user_id, "token": f"tok_{user_id}_{uuid.uuid4().hex[:8]}"}


# ========== Conversations ==========

@app.get("/api/conversations/{user_id}")
def list_conversations(user_id: str):
    conv_dir = os.path.join(CONVERSATIONS_DIR, user_id)
    os.makedirs(conv_dir, exist_ok=True)
    convs = []
    for fname in sorted(os.listdir(conv_dir), reverse=True):
        if fname.endswith(".json"):
            fpath = os.path.join(conv_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    c = json.load(f)
                convs.append({
                    "id": c.get("id"),
                    "title": c.get("title", "New Chat"),
                    "created_at": c.get("created_at"),
                    "message_count": len(c.get("messages", []))
                })
            except Exception:
                pass
    return {"conversations": convs}


@app.post("/api/conversations/{user_id}")
def create_conversation(user_id: str, body: dict = None):
    conv_id = uuid.uuid4().hex[:12]
    conv = {
        "id": conv_id, "user_id": user_id,
        "title": (body or {}).get("title", "New Chat"),
        "created_at": datetime.now().isoformat(), "messages": []
    }
    conv_dir = os.path.join(CONVERSATIONS_DIR, user_id)
    os.makedirs(conv_dir, exist_ok=True)
    _save_conversation(conv_dir, conv)
    return conv


@app.get("/api/conversations/{user_id}/{conv_id}")
def get_conversation(user_id: str, conv_id: str):
    fpath = os.path.join(CONVERSATIONS_DIR, user_id, f"{conv_id}.json")
    if not os.path.exists(fpath):
        raise HTTPException(404, "Conversation not found")
    with open(fpath, "r", encoding="utf-8") as f:
        return json.load(f)


@app.delete("/api/conversations/{user_id}/{conv_id}")
def delete_conversation(user_id: str, conv_id: str):
    fpath = os.path.join(CONVERSATIONS_DIR, user_id, f"{conv_id}.json")
    if os.path.exists(fpath):
        os.remove(fpath)
    return {"ok": True}


# ========== Main Ask ==========

@app.post("/ask")
def ask_medical(data: PatientQuery, request: Request):
    t0 = time.time()
    answer, recommended_drugs, thinking = medical_agent_with_kg(
        data.patient_id, data.question, data.current_medications
    )
    total_time = time.time() - t0

    report_file = generate_treatment_table_pdf(
        data.patient_id, data.patient_name, data.question, answer, recommended_drugs
    )

    # Record metrics
    record_rag_retrieval(total_time * 0.3)  # rough estimate

    # Save conversation
    if data.conversation_id:
        conv_dir = os.path.join(CONVERSATIONS_DIR, data.patient_id)
        os.makedirs(conv_dir, exist_ok=True)
        fpath = os.path.join(conv_dir, f"{data.conversation_id}.json")
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                conv = json.load(f)
            conv["messages"].append({"role": "user", "content": data.question})
            conv["messages"].append({"role": "agent", "content": answer, "report_file": report_file})
            if not conv.get("title") or conv["title"] == "New Chat":
                conv["title"] = data.question[:30]
            _save_conversation(conv_dir, conv)

    return {
        "answer": answer,
        "report_file": report_file,
        "thinking": thinking,
        "model": get_model_config_for_display(),
        "elapsed_ms": int(total_time * 1000),
    }


# ========== Knowledge Graph Visualization ==========

@app.get("/api/graph/visualize")
def visualize_graph(diseases: str = "", max_nodes: int = 60):
    from knowledge_graph import get_knowledge_graph
    disease_list = [d.strip() for d in diseases.split(",") if d.strip()]
    if not disease_list:
        return {"error": "No disease names provided", "image": None, "node_count": 0}
    kg = get_knowledge_graph()
    img_b64, node_count = kg.visualize_subgraph(disease_list, max_nodes)
    return {"image": f"data:image/png;base64,{img_b64}" if img_b64 else None,
            "node_count": node_count}


# ========== OCR ==========

@app.post("/api/ocr")
def ocr_medical_report(data: OCRRequest):
    import base64 as b64, logging
    logger = logging.getLogger(__name__)

    image_path = data.image_path
    if not os.path.exists(image_path):
        raise HTTPException(404, "Image not found")

    with open(image_path, "rb") as f:
        img_bytes = f.read()
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                ".gif": "image/gif", ".webp": "image/webp"}
    mime = mime_map.get(ext, "image/png")
    img_b64 = b64.b64encode(img_bytes).decode()

    ocr_text = None
    try:
        import pytesseract
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(img_bytes))
        ocr_text = pytesseract.image_to_string(img, lang='chi_sim+eng')
        if not ocr_text or len(ocr_text.strip()) < 20:
            ocr_text = None
    except Exception as e:
        logger.warning(f"Tesseract OCR failed: {e}")

    if not ocr_text:
        try:
            from openai import OpenAI
            cfg = get_model_config_for_display()
            from models import get_llm_config
            llm_cfg = get_llm_config()
            client = OpenAI(api_key=llm_cfg["api_key"], base_url=llm_cfg["api_base"])
            response = client.chat.completions.create(
                model=llm_cfg["model"],
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请识别并提取这张医疗报告中的所有文字，包括检测指标和数值。"},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}}
                    ]
                }], temperature=0.1, max_tokens=2000)
            ocr_text = response.choices[0].message.content
        except Exception as e:
            logger.warning(f"Vision API failed: {e}")

    if not ocr_text:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(img_bytes))
        w, h = img.size
        return {"text": f"（图片已接收：{w}x{h}px）\n\nOCR 需要安装 Tesseract。请手动输入关键指标。",
                "image_path": image_path, "ocr_available": False}

    # Format with LLM
    from models import create_chat_model
    prompt = f"""以下是从医疗报告 OCR 提取的文字，请整理分析：

{ocr_text[:3000]}

按格式输出：**关键指标**（表格）、**异常项汇总**、**诊断建议**。"""
    try:
        llm = create_chat_model(temperature=0.1)
        formatted = llm.invoke(prompt).content
    except Exception:
        formatted = ocr_text

    return {"text": formatted, "image_path": image_path, "ocr_available": True}


# ========== Image Upload ==========

@app.post("/api/upload_image")
async def upload_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files accepted")
    ext = os.path.splitext(file.filename or "img.png")[1] or ".png"
    save_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join("data/uploads", save_name)
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)
    b64 = base64.b64encode(content).decode()
    return {
        "filename": save_name, "path": save_path,
        "preview": f"data:{file.content_type};base64,{b64}", "size": len(content)
    }


# ========== PDF ==========

@app.get("/download_report")
def download_report(file_path: str):
    if not os.path.exists(file_path):
        return {"error": "File not found"}
    return FileResponse(path=file_path, filename=file_path, media_type='application/pdf')


# ========== Helpers ==========

def _save_conversation(conv_dir, conv):
    fpath = os.path.join(conv_dir, f"{conv['id']}.json")
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(conv, f, ensure_ascii=False, indent=2)


# Serve static files last
app.mount("/", StaticFiles(directory="static", html=True), name="static")
