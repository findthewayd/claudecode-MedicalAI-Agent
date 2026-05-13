# 🏥 医疗智能助手 — Medical AI Agent

基于 **LangChain + RAG + Neo4j 知识图谱** 的全栈医疗智能助手，支持多模型切换、OCR 报告识别、知识图谱可视化、药物相互作用分级检查、PDF 报告生成，Docker 一键部署。

---

## ✨ 功能特性

### 核心能力

| 功能 | 说明 |
|------|------|
| 🧠 **多模型支持** | DeepSeek / 智谱 GLM / 讯飞星火 / OpenAI 可配置切换 |
| 📚 **RAG 检索** | ChromaDB 向量库（398K 条华佗 QA），BGE 中文嵌入，FAISS 回退 |
| 🕸️ **知识图谱** | Neo4j Aura 云端 / NetworkX 内存图双后端，自动故障切换 |
| 📊 **图谱可视化** | 诊断结果自动生成彩色节点网络图（疾病/症状/药物） |
| 🔍 **症状语义匹配** | BGE 向量模型识别同义词（"浑身没劲"→"乏力"） |
| 💊 **药物交互分级** | 277 种药物，三级严重程度：🚫禁用 / ⚠️慎用 / 📝注意 |
| 📄 **PDF 报告** | 中文治疗方案报告，含药物表格 |

### 多模态 & 交互

| 功能 | 说明 |
|------|------|
| 🖼️ **图片粘贴** | Ctrl+V 直接粘贴剪贴板截图 |
| 📋 **OCR 识别** | 自动识别血常规、CT 等检查报告中的关键指标 |
| 💬 **对话管理** | 多轮对话持久化，左侧历史栏切换/删除 |
| 📝 **思考链展示** | 前端展示 Agent 完整推理过程 |
| 👤 **用户系统** | 多用户隔离，各自独立对话历史 |

### 运维 & 监控

| 功能 | 说明 |
|------|------|
| 🚪 **API 网关** | IP 级限流（可配）、请求日志、耗时追踪 |
| 📈 **Prometheus 指标** | 接口延迟(P95/P99)、成功率、LLM/RAG 耗时、内存占用 |
| 📊 **Grafana 看板** | 预配置数据源，开箱即用 |
| 🐳 **Docker 部署** | 一键 `docker-compose up`（含 Tesseract 中文 OCR） |
| 📝 **结构化日志** | JSON 格式，患者 ID 自动脱敏 |

---

## 📁 项目结构

```
d:/yiliao/
├── server.py                  # FastAPI 后端（API 入口 + 中间件）
├── agent.py                   # 核心智能体：RAG + KG + 症状提取 + 药物检查
├── models.py                  # 多模型工厂（DeepSeek/智谱/星火/OpenAI）
├── knowledge_graph.py         # 知识图谱（NetworkX + Neo4j 双后端 + 可视化）
├── memory.py                  # 患者对话记忆（pickle 持久化）
├── middleware.py              # API 网关（限流 + 鉴权 + Prometheus 指标）
├── logging_config.py          # 结构化 JSON 日志 + PII 脱敏
├── requirements.txt           # Python 依赖
├── .env                       # 环境变量
├── Dockerfile                 # Docker 镜像
├── docker-compose.yml         # 一键部署（App + Prometheus + Grafana）
├── prometheus.yml             # Prometheus 采集配置
├── grafana-datasources.yml    # Grafana 数据源
│
├── static/
│   └── index.html             # 前端 SPA（登录/对话/图片/图谱/PDF）
│
├── data/
│   ├── pubmed_articles.txt    # 医疗文本语料（8,808 条疾病描述）
│   ├── drug_interactions.json # 药物相互作用数据库（277 种，含分级）
│   ├── knowledge_graph.pkl    # NetworkX 预构建图谱
│   ├── conversations/         # 用户对话历史（JSON）
│   └── uploads/               # 用户上传的图片
│
├── huatuo_data/
│   ├── medical1.json          # 8,808 种疾病结构化数据（JSONL）
│   ├── chroma_huatuo_db/      # Chroma 向量数据库（398K 条 QA）
│   ├── BAAI/bge-small-zh-v1.5/# 本地 BGE 中文嵌入模型
│   └── industry_instruction_semantic_cluster_dedup_医疗_train.jsonl
│                               # 348K 条医疗对话训练数据
│
├── scripts/
│   ├── build_pubmed_corpus.py       # 生成 PubMed 语料
│   ├── build_drug_interactions.py   # 生成药物交互数据库
│   └── build_knowledge_graph.py     # 构建 NetworkX 图谱
│
├── db/
│   └── patient_history.pkl    # 患者对话记录
│
├── embaddignxiazai.py         # 下载 BGE 嵌入模型
├── 数据集下载.py               # 下载华佗数据集
└── wenjianqianyi.py           # 迁移数据集缓存
```

---

## 🚀 快速开始

### 首次克隆后构建

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 .env（填入 API Key）
cp .env.example .env

# 3. 构建数据文件（必需）
python scripts/build_drug_interactions.py       # data/drug_interactions.json (88KB)
python scripts/build_pubmed_corpus.py           # data/pubmed_articles.txt (25MB)
python scripts/build_knowledge_graph.py         # data/knowledge_graph.pkl (46MB)

# 4. 启动
uvicorn server:app --reload

# 5. 浏览器打开 http://127.0.0.1:8000
```

### 方式一：本地运行

```bash
pip install -r requirements.txt
cp .env.example .env   # 编辑填入 API Key
# 按上方「首次克隆后构建」步骤生成数据文件
uvicorn server:app --reload
```

### 方式二：Docker 一键部署

```bash
docker-compose up -d
```

服务端口：

| 服务 | 地址 | 说明 |
|------|------|------|
| 医疗助手 | http://localhost:8000 | 主应用 |
| Grafana | http://localhost:3000 | 监控看板（admin/admin） |
| Prometheus | http://localhost:9090 | 指标采集 |

---

## ⚙️ 配置说明

### 模型切换

修改 `.env` 中 `LLM_PROVIDER` 即可切换大模型：

```env
# 使用 DeepSeek（默认）
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-your-key
DEEPSEEK_MODEL=deepseek-chat

# 使用智谱 GLM
LLM_PROVIDER=zhipu
ZHIPU_API_KEY=your-zhipu-key
ZHIPU_MODEL=glm-4-plus

# 使用讯飞星火
LLM_PROVIDER=spark
SPARK_API_KEY=your-spark-key
SPARK_API_SECRET=your-spark-secret
SPARK_APP_ID=your-app-id
SPARK_MODEL=spark-lite

# 使用 OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key
OPENAI_MODEL=gpt-4o
```

### Neo4j（可选）

不配置则自动使用本地 NetworkX 内存图谱：

```env
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=your-username
NEO4J_PASSWORD=your-password
```

首次连接 Neo4j 空数据库会自动导入 8,808 种疾病数据。

### 限流参数

```env
RATE_LIMIT_MAX=30       # 每窗口最大请求数
RATE_LIMIT_WINDOW=60    # 窗口时长（秒）
```

---

## 🔧 技术架构

```
┌─ 浏览器 (static/index.html) ───────────────────────────────┐
│  • 登录 / 对话管理 / 图片粘贴 / OCR / 图谱展示 / PDF 下载  │
└──────────────────────────┬─────────────────────────────────┘
                           │ POST /ask
                           ▼
┌─ API 网关 (middleware.py) ──────────────────────────────────┐
│  • 限流 / 请求日志 / Prometheus 指标采集                    │
└──────────────────────────┬─────────────────────────────────┘
                           ▼
┌─ FastAPI (server.py) ──────────────────────────────────────┐
│  /ask  /api/ocr  /api/graph/visualize  /metrics            │
└──────────────────────────┬─────────────────────────────────┘
                           ▼
┌─ medical_agent_with_kg() (agent.py) ────────────────────────┐
│                                                            │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────┐ │
│  │ RAG 检索     │  │ 知识图谱分析     │  │ 模型调用     │ │
│  │              │  │                  │  │              │ │
│  │ ChromaDB ◀──┤  │ Neo4j / NetworkX │  │ models.py    │ │
│  │ (398K QA)    │  │ (8808 疾病)      │  │ DeepSeek     │ │
│  │              │  │                  │  │ 智谱 / 星火  │ │
│  │ FAISS (回退) │  │ 症状语义匹配     │  │ OpenAI       │ │
│  └──────────────┘  └──────────────────┘  └──────────────┘ │
│                                                            │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────┐ │
│  │ 药物检查     │  │ PDF 报告         │  │ 对话记忆     │ │
│  │              │  │                  │  │              │ │
│  │ 277 种药物   │  │ fpdf2            │  │ PatientMemory│ │
│  │ 三级分级     │  │ 中文 SimHei      │  │ pickle       │ │
│  └──────────────┘  └──────────────────┘  └──────────────┘ │
└────────────────────────────────────────────────────────────┘
```

### 知识图谱 Schema

```
节点类型:  Disease  │  Symptom  │  Drug  │  Department  │  Check  │  Food
─────────────────────────────────────────────────────────────────────────
关系类型:
  (Disease)-[:HAS_SYMPTOM]→(Symptom)      症状关联
  (Disease)-[:TREATS]→(Drug)              推荐药物
  (Disease)-[:HAS_COMPLICATION]→(Disease)  并发症
  (Disease)-[:TREATED_IN]→(Department)     就诊科室
  (Disease)-[:CHECKED_BY]→(Check)          检查项目
  (Disease)-[:CURED_BY]→(CureWay)          治疗方式
  (Disease)-[:BELONGS_TO]→(Category)       疾病分类
  (Disease)-[:DO_EAT|NOT_EAT|RECOMMEND_EAT]→(Food)  饮食建议
```

数据来源：`huatuo_data/medical1.json`（8,808 疾病 · 25,000+ 症状 · 21,000+ 药物）

---

## 📡 API 文档

### `POST /ask` — 核心问诊

```json
// Request
{
  "patient_id": "user1",
  "patient_name": "张三",
  "question": "咳嗽发烧三天，浑身没劲",
  "current_medications": ["阿莫西林"],
  "conversation_id": "abc123"
}

// Response
{
  "answer": "1. 可能诊断：...\n2. 风险评估：...",
  "report_file": "treatment_report_user1_20260512.pdf",
  "thinking": [
    "正在加载知识图谱...",
    "知识图谱已就绪（8807 种疾病）",
    "识别到症状: 咳嗽、发热、乏力",
    "正在检索医疗知识库...",
    "图谱分析完成，匹配到 5 种可能疾病",
    "药物相互作用检查完成",
    "分析完成"
  ],
  "model": {"provider": "deepseek", "model": "deepseek-chat"},
  "elapsed_ms": 3200
}
```

### 其他接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/login` | 用户登录 `{username, password}` |
| `GET` | `/api/conversations/{user_id}` | 获取对话列表 |
| `POST` | `/api/conversations/{user_id}` | 创建新对话 |
| `DELETE` | `/api/conversations/{user_id}/{conv_id}` | 删除对话 |
| `POST` | `/api/upload_image` | 上传图片（返回 base64 预览） |
| `POST` | `/api/ocr` | OCR 识别报告 `{image_path}` |
| `GET` | `/api/graph/visualize?diseases=百日咳,肺炎` | 知识图谱可视化 |
| `GET` | `/download_report?file_path=xxx.pdf` | 下载 PDF 报告 |
| `GET` | `/api/health` | 健康检查 + 当前模型信息 |
| `GET` | `/metrics` | Prometheus 指标 |

---

## 📊 监控指标

### Prometheus 指标

| 指标名 | 类型 | 说明 |
|--------|------|------|
| `http_requests_total` | Counter | 请求计数（method/endpoint/status） |
| `http_request_duration_seconds` | Histogram | 接口延迟分布（P50/P95/P99） |
| `llm_call_duration_seconds` | Histogram | LLM 调用耗时（provider/model） |
| `llm_requests_total` | Counter | LLM 调用计数（success/error） |
| `rag_retrieval_duration_seconds` | Histogram | RAG 检索耗时 |
| `kg_queries_total` | Counter | 知识图谱查询次数 |
| `app_memory_bytes` | Gauge | 应用内存占用 |
| `app_active_requests` | Gauge | 当前活跃请求数 |

### Grafana

`docker-compose up` 后访问 `http://localhost:3000`（admin/admin），Prometheus 数据源已预配置。

---

## 🐳 Docker 部署

```bash
# 构建并启动全部服务
docker-compose up -d --build

# 查看日志
docker-compose logs -f app

# 停止
docker-compose down

# 清理数据卷
docker-compose down -v
```

Dockerfile 已预装 Tesseract OCR 中文语言包，OCR 功能开箱可用。

---

## ⚠️ 注意事项

| 项 | 说明 |
|----|------|
| API Key | `.env` 中必须填写至少一个模型的 API Key |
| BGE 模型 | 首次请求加载 ~200MB 嵌入模型，后续复用缓存 |
| Neo4j | 不配置则自动回退 NetworkX，不影响使用 |
| Tesseract | Windows 本地需手动安装 Tesseract 才能 OCR；Docker 已内置 |
| 中文字体 | PDF 使用 SimHei 字体，Windows 系统自带 |
| 限流 | 默认 30 次/分钟，可通过 `.env` 调整 |
| 隐私 | 日志自动对患者 ID 脱敏（保留前 2 位 + hash） |

---

## 📄 License

MIT
