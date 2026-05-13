import os
import pickle
import datetime
import json
import logging
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from models import create_chat_model
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from memory import PatientMemory
from knowledge_graph import get_knowledge_graph

# Chinese font paths for fpdf2 (cross-platform detection)
_CJK_FONT_CANDIDATES = [
    "C:/Windows/Fonts/simhei.ttf",           # Windows
    "C:/Windows/Fonts/msyh.ttc",             # Windows fallback
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # Linux
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux (Noto)
    "/System/Library/Fonts/PingFang.ttc",    # macOS
    "/System/Library/Fonts/STHeiti Light.ttc", # macOS fallback
]
_CJK_FONT_PATH = None
for _p in _CJK_FONT_CANDIDATES:
    if os.path.exists(_p):
        _CJK_FONT_PATH = _p
        break
if not _CJK_FONT_PATH:
    _CJK_FONT_PATH = "Helvetica"  # no CJK font found — PDF won't show Chinese

load_dotenv()
memory = PatientMemory()

_kg = None


def get_kg():
    global _kg
    if _kg is None:
        _kg = get_knowledge_graph()
    return _kg

logger = logging.getLogger(__name__)

# Paths
CHROMA_PERSIST_DIR = "huatuo_data/chroma_huatuo_db"
EMBED_MODEL_PATH = "huatuo_data/BAAI/bge-small-zh-v1.5"
FAISS_PKL = "vectorstore.pkl"
PUBMED_FILE = "data/pubmed_articles.txt"
DRUG_DB_FILE = "data/drug_interactions.json"

_retriever = None


def get_chroma_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_PATH)
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="huatuo_qa_rag"
    )
    return vectorstore.as_retriever(search_kwargs={"k": 5})


def get_faiss_retriever():
    if os.path.exists(FAISS_PKL):
        with open(FAISS_PKL, "rb") as f:
            vectorstore = pickle.load(f)
    else:
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(PUBMED_FILE, encoding="utf-8")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        docs = splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_PATH)
        vectorstore = FAISS.from_documents(docs, embeddings)
        with open(FAISS_PKL, "wb") as f:
            pickle.dump(vectorstore, f)
    return vectorstore.as_retriever()


def get_retriever():
    global _retriever
    if _retriever is not None:
        return _retriever
    try:
        _retriever = get_chroma_retriever()
        logger.info("Using Chroma retriever (huatuo_encyclopedia_qa)")
    except Exception as e:
        logger.warning(f"Chroma unavailable, falling back to FAISS: {e}")
        _retriever = get_faiss_retriever()
    return _retriever


# ===== Drug database =====
_drug_db = None


def get_drug_db():
    global _drug_db
    if _drug_db is not None:
        return _drug_db
    if os.path.exists(DRUG_DB_FILE):
        with open(DRUG_DB_FILE, "r", encoding="utf-8") as f:
            _drug_db = json.load(f)
    else:
        _drug_db = {}
    return _drug_db


SEVERITY_LABELS = {"禁用": "🚫 禁止合用", "慎用": "⚠️ 谨慎使用", "注意": "📝 需加注意"}


def check_drug_interactions(prescribed_drugs, new_drugs):
    """Check drug interactions. Returns list of (drug1, drug2, severity)."""
    drug_db = get_drug_db()
    conflicts = []
    for new_drug in new_drugs:
        interacting = drug_db.get(new_drug, {})
        # Support both old list format and new dict format
        if isinstance(interacting, list):
            for pd in prescribed_drugs:
                if pd in interacting:
                    conflicts.append((new_drug, pd, "注意"))
        else:
            for pd in prescribed_drugs:
                if pd in interacting:
                    conflicts.append((new_drug, pd, interacting[pd]))
    return conflicts


# ===== Agent core =====
def medical_agent(patient_id, query):
    history = memory.get_history(patient_id)
    history_text = "\n".join([f"{h['role']}: {h['content']}" for h in history])
    prompt = f"""
你是专业医疗智能助手。
患者历史信息：
{history_text}

当前问题：
{query}

请给出：
1. 可能诊断
2. 风险评估
3. 治疗建议（标记药物为 [Drug]:剂量）
4. 提供可参考文献（PubMed编号）
输出要结构化。
"""
    llm = create_chat_model(temperature=0.2)
    retriever = get_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    answer = qa.run(prompt)

    memory.add_message(patient_id, "user", query)
    memory.add_message(patient_id, "agent", answer)
    return answer


# ===== Knowledge Graph enhanced agent =====

import numpy as np

# Symptom embedding cache
_symptom_embeddings = None
_symptom_names = None
_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_PATH)
    return _embed_model


def _build_symptom_index():
    """Pre-compute embeddings for all KG symptom names."""
    global _symptom_embeddings, _symptom_names
    if _symptom_embeddings is not None:
        return
    kg = get_kg()
    if hasattr(kg, '_symptom_index'):
        _symptom_names = list(kg._symptom_index.keys())
    else:
        _symptom_names = []
    if not _symptom_names:
        _symptom_embeddings = np.array([])
        return
    model = _get_embed_model()
    vectors = model.embed_documents(_symptom_names)
    _symptom_embeddings = np.array(vectors)
    logger.info(f"Built symptom index: {len(_symptom_names)} symptoms")


def _cosine_similarity(a, b):
    """Row-wise cosine similarity between vector a and matrix b."""
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return np.dot(b_norm, a_norm)


def extract_symptoms_from_query(query: str, top_k: int = 8, min_score: float = 0.55) -> list[str]:
    """Extract symptoms using exact matching + semantic similarity."""
    kg = get_kg()
    found = set()

    # Stage 1: Exact/fuzzy matching (fast)
    if hasattr(kg, '_symptom_index'):
        for symptom in kg._symptom_index:
            if symptom in query or query in symptom:
                found.add(symptom)

    # Stage 2: Semantic matching for synonyms
    if len(found) < 3:  # Only do expensive embedding search if few exact matches
        try:
            _build_symptom_index()
            if _symptom_embeddings is not None and len(_symptom_embeddings) > 0:
                model = _get_embed_model()
                query_vec = np.array(model.embed_query(query))
                scores = _cosine_similarity(query_vec, _symptom_embeddings)
                top_indices = np.argsort(scores)[::-1][:top_k]
                for idx in top_indices:
                    if scores[idx] >= min_score:
                        found.add(_symptom_names[idx])
        except Exception as e:
            logger.warning(f"Semantic symptom extraction failed: {e}")

    return list(found)


def medical_agent_with_kg(patient_id, query, current_medications=None):
    """Enhanced medical agent that combines RAG + Knowledge Graph.
    Returns (answer, recommended_drugs, thinking_steps)."""
    if current_medications is None:
        current_medications = []
    thinking = []

    # Step 1: Load KG
    thinking.append("正在加载知识图谱...")
    try:
        kg = get_kg()
        thinking.append(f"知识图谱已就绪（{kg.get_disease_count()} 种疾病）")
    except Exception:
        kg = None
        thinking.append("知识图谱加载失败，将仅使用 RAG 检索")

    # Step 2: Extract symptoms
    thinking.append("正在从提问中提取症状...")
    symptoms = extract_symptoms_from_query(query)
    if symptoms:
        thinking.append(f"识别到症状: {'、'.join(symptoms[:8])}")
    else:
        thinking.append("未识别到明确症状关键词，将使用语义匹配...")

    # Step 3: RAG retrieval
    thinking.append("正在检索医疗知识库（ChromaDB）...")

    # Get base RAG answer
    answer = medical_agent(patient_id, query)
    thinking.append("DeepSeek 已生成回答")

    # Step 4: KG differential diagnosis
    if kg and symptoms:
        thinking.append("正在进行知识图谱诊断分析...")
        diagnoses = kg.get_diseases_by_symptoms(symptoms, top_k=5)
        if diagnoses:
            kg_section = "\n\n## 知识图谱分析\n\n**可能的疾病（基于症状匹配）：**\n"
            for disease, score in diagnoses:
                kg_section += f"- {disease}（匹配度: {score}）\n"
                if score >= 1.0:
                    drugs = kg.get_drugs_for_disease(disease)
                    if drugs:
                        kg_section += f"  推荐药物: {'、'.join(drugs[:5])}\n"
            thinking.append(f"图谱分析完成，匹配到 {len(diagnoses)} 种可能疾病")

            # Step 5: Drug interaction check
            recommended_drugs = []
            for line in answer.split("\n"):
                if "[Drug]" in line:
                    parts = line.split(":")
                    recommended_drugs.append(parts[0].replace("[Drug]", "").strip())

            all_meds = current_medications + recommended_drugs
            if len(all_meds) > 1:
                kg_section += "\n**药物安全性提示：**\n"
                all_conflicts = check_drug_interactions(all_meds, all_meds)
                found_conflict = False
                warned_pairs = set()
                if all_conflicts:
                    all_conflicts.sort(key=lambda x: {"禁用": 3, "慎用": 2, "注意": 1}.get(x[2], 0), reverse=True)
                    for d1, d2, sev in all_conflicts:
                        pair = tuple(sorted([d1, d2]))
                        if pair in warned_pairs:
                            continue
                        warned_pairs.add(pair)
                        label = SEVERITY_LABELS.get(sev, f"⚠️ {sev}")
                        kg_section += f"- {label}: {d1} 与 {d2}\n"
                        found_conflict = True
                if not found_conflict:
                    kg_section += "- 未发现已知药物相互作用\n"
                thinking.append("药物相互作用检查完成")

            answer += kg_section
        else:
            recommended_drugs = []
            thinking.append("未匹配到高相关度疾病")
    else:
        recommended_drugs = []

    thinking.append("分析完成")
    return answer, recommended_drugs, thinking


# ===== PDF generation =====
class MedicalPDF(FPDF):
    def __init__(self):
        super().__init__()
        if os.path.exists(_CJK_FONT_PATH):
            self.add_font("CJK", "", _CJK_FONT_PATH)
            self._font_name = "CJK"
        else:
            self._font_name = "Helvetica"

    def header(self):
        if self._font_name != "Helvetica":
            self.set_font(self._font_name, "", 14)
        else:
            self.set_font(self._font_name, "", 14)
        self.cell(0, 10, "Medical Agent Treatment Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.ln(5)


def generate_treatment_table_pdf(patient_id, patient_name, question, answer, recommended_drugs):
    file_name = f"treatment_report_{patient_id}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    pdf = MedicalPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    font = pdf._font_name

    # Patient info
    pdf.set_font(font, "", 11)
    pdf.cell(0, 7, f"Patient ID: {patient_id}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 7, f"Patient Name: {patient_name}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 7, f"Question: {question}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)

    # Drug table
    if recommended_drugs:
        pdf.set_font(font, "", 10)
        pdf.set_fill_color(173, 216, 230)  # lightblue
        col_w = [60, 60, 70]
        headers = ["Drug", "Dosage", "Notes"]
        for i, h in enumerate(headers):
            pdf.cell(col_w[i], 8, h, border=1, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.ln()
        for drug in recommended_drugs:
            pdf.cell(col_w[0], 7, drug, border=1)
            pdf.cell(col_w[1], 7, "Follow guidelines", border=1)
            pdf.cell(col_w[2], 7, "Review patient history", border=1)
            pdf.ln()
        pdf.ln(5)

    # Answer text
    pdf.set_font(font, "", 10)
    for line in answer.split("\n"):
        if line.strip():
            pdf.cell(0, 6, line.strip(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:
            pdf.ln(4)

    pdf.output(file_name)
    return file_name
