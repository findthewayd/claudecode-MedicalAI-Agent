"""Knowledge graph module for medical intelligence agent.

Provides two backends:
- NetworkXBackend: In-memory graph, works out of the box
- Neo4jBackend: Neo4j graph database, requires Neo4j server

Unified via KnowledgeGraphBackend abstract interface.
"""
import json
import os
import pickle
import logging
from abc import ABC, abstractmethod
from collections import defaultdict

import networkx as nx
import io
import base64

logger = logging.getLogger(__name__)

# Try to setup matplotlib with Chinese font
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    _CJK_FONT = None
    _font_candidates = [
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
    ]
    for _fp in _font_candidates:
        if os.path.exists(_fp):
            _CJK_FONT = FontProperties(fname=_fp)
            break
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def render_subgraph_image(disease_names, symptom_map, drug_map, complication_map,
                          max_nodes=60):
    """Render a subgraph as base64 PNG. Returns (base64_str, node_count)."""
    if not _HAS_MPL:
        return None, 0

    G = nx.Graph()
    added_diseases = set()

    for d in disease_names[:10]:
        if d not in added_diseases:
            G.add_node(d, ntype='disease')
            added_diseases.add(d)

        for s in symptom_map.get(d, [])[:4]:
            G.add_node(s, ntype='symptom')
            G.add_edge(d, s, rel='症状')
            if G.number_of_nodes() >= max_nodes:
                break

        for dr in drug_map.get(d, [])[:3]:
            G.add_node(dr, ntype='drug')
            G.add_edge(d, dr, rel='药物')
            if G.number_of_nodes() >= max_nodes:
                break

        for comp in complication_map.get(d, [])[:2]:
            G.add_node(comp, ntype='disease')
            G.add_edge(d, comp, rel='并发症')
            added_diseases.add(comp)
            if G.number_of_nodes() >= max_nodes:
                break

        if G.number_of_nodes() >= max_nodes:
            break

    if G.number_of_nodes() == 0:
        return None, 0

    # Render
    fig, ax = plt.subplots(figsize=(14, 10))
    color_map = {'disease': '#e74c3c', 'symptom': '#3498db', 'drug': '#2ecc71'}
    node_colors = [color_map.get(G.nodes[n].get('ntype', ''), '#95a5a6') for n in G.nodes()]
    node_sizes = [800 if G.nodes[n].get('ntype') == 'disease' else 500 for n in G.nodes()]

    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='#888', ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                           alpha=0.9, ax=ax)

    # Labels with Chinese font
    for node, (x, y) in pos.items():
        label = node if len(node) <= 8 else node[:7] + '…'
        if _CJK_FONT:
            ax.text(x, y, label, fontproperties=_CJK_FONT, fontsize=11,
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              alpha=0.7, edgecolor='none'))
        else:
            ax.text(x, y, label, fontsize=9, ha='center', va='center')

    # Legend
    legend_items = [('疾病', '#e74c3c'), ('症状', '#3498db'), ('药物', '#2ecc71')]
    for i, (label, c) in enumerate(legend_items):
        ax.plot([], [], 'o', color=c, label=label, markersize=10)
    if _CJK_FONT:
        ax.legend(prop=_CJK_FONT, loc='upper right', fontsize=12)
    else:
        ax.legend(loc='upper right', fontsize=12)

    ax.set_title('知识图谱子图', fontproperties=_CJK_FONT if _CJK_FONT else None,
                 fontsize=18, pad=15)
    ax.axis('off')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode(), G.number_of_nodes()

# ========================
# Abstract Backend
# ========================


class KnowledgeGraphBackend(ABC):
    @abstractmethod
    def get_disease_info(self, disease_name: str) -> dict | None:
        """Get all information about a disease."""

    @abstractmethod
    def get_diseases_by_symptoms(self, symptoms: list[str], top_k: int = 10) -> list[tuple[str, int]]:
        """Find diseases matching given symptoms. Returns [(disease_name, match_count), ...]."""

    @abstractmethod
    def get_drugs_for_disease(self, disease_name: str) -> list[str]:
        """Get recommended drugs for a disease."""

    @abstractmethod
    def get_symptoms_for_disease(self, disease_name: str) -> list[str]:
        """Get symptoms associated with a disease."""

    @abstractmethod
    def get_diseases_by_department(self, department: str) -> list[str]:
        """Get diseases treated in a department."""

    @abstractmethod
    def search_diseases(self, keyword: str, top_k: int = 10) -> list[str]:
        """Search diseases by keyword in name."""

    @abstractmethod
    def visualize_subgraph(self, disease_names: list[str], max_nodes: int = 60) -> tuple[str | None, int]:
        """Render a subgraph image for given diseases. Returns (base64_png, node_count)."""

    @abstractmethod
    def get_disease_count(self) -> int:
        """Get total number of diseases in the graph."""


# ========================
# NetworkX Backend
# ========================


class NetworkXBackend(KnowledgeGraphBackend):
    def __init__(self, medical_json_path: str = None, pickle_path: str = None):
        self.graph = nx.MultiDiGraph()
        self._symptom_index = defaultdict(set)
        self._drug_index = defaultdict(set)
        self._dept_index = defaultdict(set)
        self._disease_names = set()

        if pickle_path and os.path.exists(pickle_path):
            self._load_pickle(pickle_path)
        elif medical_json_path and os.path.exists(medical_json_path):
            self._build_from_json(medical_json_path)

    def _add_node(self, node_type: str, name: str, **properties):
        node_id = f"{node_type}:{name}"
        if not self.graph.has_node(node_id):
            self.graph.add_node(node_id, type=node_type, name=name, **properties)

    def _build_from_json(self, path: str):
        logger.info(f"Building knowledge graph from {path}...")
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue

                name = d.get("name", "")
                if not name:
                    continue

                self._disease_names.add(name)
                disease_id = f"disease:{name}"

                # Disease node
                self.graph.add_node(disease_id, type="disease",
                                    name=name, desc=d.get("desc", ""),
                                    cause=d.get("cause", ""),
                                    prevent=d.get("prevent", ""),
                                    cured_prob=d.get("cured_prob", ""),
                                    cost_money=d.get("cost_money", ""),
                                    cure_lasttime=d.get("cure_lasttime", ""),
                                    get_prob=d.get("get_prob", ""))

                # Categories
                for cat in d.get("category", []) or []:
                    self._add_node("category", cat)
                    self.graph.add_edge(disease_id, f"category:{cat}", relation="BELONGS_TO")

                # Symptoms
                for s in d.get("symptom", []) or []:
                    s = s.strip()
                    if not s:
                        continue
                    self._add_node("symptom", s)
                    self.graph.add_edge(disease_id, f"symptom:{s}", relation="HAS_SYMPTOM")
                    self._symptom_index[s].add(name)

                # Departments
                for dept in d.get("cure_department", []) or []:
                    dept = dept.strip()
                    if not dept:
                        continue
                    self._add_node("department", dept)
                    self.graph.add_edge(disease_id, f"department:{dept}", relation="TREATED_IN")
                    self._dept_index[dept].add(name)

                # Cure ways
                for cw in d.get("cure_way", []) or []:
                    cw = cw.strip()
                    if not cw:
                        continue
                    self._add_node("cureway", cw)
                    self.graph.add_edge(disease_id, f"cureway:{cw}", relation="CURED_BY")

                # Checks
                for chk in d.get("check", []) or []:
                    chk = chk.strip()
                    if not chk:
                        continue
                    self._add_node("check", chk)
                    self.graph.add_edge(disease_id, f"check:{chk}", relation="CHECKED_BY")

                # Drugs (recommended + common)
                for rd in (d.get("recommand_drug", []) or []) + (d.get("common_drug", []) or []):
                    rd = rd.strip()
                    if not rd:
                        continue
                    self._add_node("drug", rd)
                    self.graph.add_edge(disease_id, f"drug:{rd}", relation="TREATS")
                    self._drug_index[rd].add(name)

                # Complications
                for comp in d.get("acompany", []) or []:
                    comp = comp.strip()
                    if not comp:
                        continue
                    self._disease_names.add(comp)
                    comp_id = f"disease:{comp}"
                    if not self.graph.has_node(comp_id):
                        self.graph.add_node(comp_id, type="disease", name=comp)
                    self.graph.add_edge(disease_id, comp_id, relation="HAS_COMPLICATION")

                # Diet
                for food in d.get("do_eat", []) or []:
                    food = food.strip()
                    if not food:
                        continue
                    self._add_node("food", food)
                    self.graph.add_edge(disease_id, f"food:{food}", relation="DO_EAT")
                for food in d.get("not_eat", []) or []:
                    food = food.strip()
                    if not food:
                        continue
                    self._add_node("food", food)
                    self.graph.add_edge(disease_id, f"food:{food}", relation="NOT_EAT")
                for food in d.get("recommand_eat", []) or []:
                    food = food.strip()
                    if not food:
                        continue
                    self._add_node("food", food)
                    self.graph.add_edge(disease_id, f"food:{food}", relation="RECOMMEND_EAT")

                count += 1

        logger.info(f"Built graph with {count} diseases, {self.graph.number_of_nodes()} nodes, "
                     f"{self.graph.number_of_edges()} edges")

    def save_pickle(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = (self.graph, dict(self._symptom_index), dict(self._drug_index),
                dict(self._dept_index), self._disease_names)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved knowledge graph to {path}")

    def _load_pickle(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.graph = data[0]
        self._symptom_index = defaultdict(set, data[1])
        self._drug_index = defaultdict(set, data[2])
        self._dept_index = defaultdict(set, data[3])
        self._disease_names = data[4]
        logger.info(f"Loaded knowledge graph: {self.graph.number_of_nodes()} nodes, "
                     f"{len(self._disease_names)} diseases")

    # ---- Query methods ----

    def get_disease_info(self, disease_name: str) -> dict | None:
        disease_id = f"disease:{disease_name}"
        if not self.graph.has_node(disease_id):
            return None
        node = self.graph.nodes[disease_id]
        info = dict(node)
        info["symptoms"] = self.get_symptoms_for_disease(disease_name)
        info["drugs"] = self.get_drugs_for_disease(disease_name)
        info["complications"] = self._get_neighbor_names(disease_id, "HAS_COMPLICATION")
        info["departments"] = self._get_neighbor_names(disease_id, "TREATED_IN")
        info["checks"] = self._get_neighbor_names(disease_id, "CHECKED_BY")
        info["cure_ways"] = self._get_neighbor_names(disease_id, "CURED_BY")
        info["do_eat"] = self._get_neighbor_names(disease_id, "DO_EAT")
        info["not_eat"] = self._get_neighbor_names(disease_id, "NOT_EAT")
        return info

    def get_diseases_by_symptoms(self, symptoms: list[str], top_k: int = 10) -> list[tuple[str, int]]:
        scores = defaultdict(int)
        for symptom in symptoms:
            for name in self._symptom_index.get(symptom, set()):
                scores[name] += 1
            # Also try partial matching
            for idx_symptom, diseases in self._symptom_index.items():
                if symptom in idx_symptom or idx_symptom in symptom:
                    for name in diseases:
                        scores[name] += 0.5
        sorted_diseases = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_diseases[:top_k]

    def get_drugs_for_disease(self, disease_name: str) -> list[str]:
        return self._get_neighbor_names(f"disease:{disease_name}", "TREATS")

    def get_symptoms_for_disease(self, disease_name: str) -> list[str]:
        return self._get_neighbor_names(f"disease:{disease_name}", "HAS_SYMPTOM")

    def get_diseases_by_department(self, department: str) -> list[str]:
        return sorted(self._dept_index.get(department, set()))

    def search_diseases(self, keyword: str, top_k: int = 10) -> list[str]:
        results = []
        for name in self._disease_names:
            if keyword in name:
                results.append(name)
                if len(results) >= top_k:
                    break
        return results

    def visualize_subgraph(self, disease_names: list[str], max_nodes: int = 60) -> tuple[str | None, int]:
        symptom_map = {}
        drug_map = {}
        comp_map = {}
        for d in disease_names:
            symptom_map[d] = self.get_symptoms_for_disease(d)
            drug_map[d] = self.get_drugs_for_disease(d)
            comp_map[d] = self._get_neighbor_names(f"disease:{d}", "HAS_COMPLICATION")
        return render_subgraph_image(disease_names, symptom_map, drug_map, comp_map, max_nodes)

    def get_disease_count(self) -> int:
        return len(self._disease_names)

    def _get_neighbor_names(self, node_id: str, relation: str) -> list[str]:
        """Get names of neighbor nodes connected by a specific relation."""
        names = []
        for _, neighbor, edge_data in self.graph.out_edges(node_id, data=True):
            if edge_data.get("relation") == relation:
                node = self.graph.nodes[neighbor]
                names.append(node.get("name", neighbor))
        return names


# ========================
# Neo4j Backend
# ========================


class Neo4jBackend(KnowledgeGraphBackend):
    def __init__(self, uri: str = None, user: str = None, password: str = None):
        from dotenv import load_dotenv
        load_dotenv()
        self.uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.environ.get("NEO4J_USERNAME") or os.environ.get("NEO4J_USER", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASSWORD", "neo4j")
        self.driver = None
        self._symptom_index = defaultdict(set)
        self._drug_index = defaultdict(set)
        self._dept_index = defaultdict(set)
        self._disease_names = set()
        self._connect()
        self._load_indexes()

    def _connect(self):
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.warning(f"Neo4j connection failed: {e}")
            raise

    def _load_indexes(self):
        """Load symptom/drug/dept indexes into memory for fast lookup."""
        try:
            records = self._query("MATCH (d:Disease) RETURN d.name AS name")
            self._disease_names = {r["name"] for r in records}

            records = self._query(
                "MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom) "
                "RETURN s.name AS symptom, collect(d.name) AS diseases")
            for r in records:
                self._symptom_index[r["symptom"]] = set(r["diseases"])

            records = self._query(
                "MATCH (d:Disease)-[:TREATS]->(dd:Drug) "
                "RETURN dd.name AS drug, collect(d.name) AS diseases")
            for r in records:
                self._drug_index[r["drug"]] = set(r["diseases"])

            records = self._query(
                "MATCH (d:Disease)-[:TREATED_IN]->(dp:Department) "
                "RETURN dp.name AS dept, collect(d.name) AS diseases")
            for r in records:
                self._dept_index[r["dept"]] = set(r["diseases"])

            logger.info(f"Loaded indexes: {len(self._symptom_index)} symptoms, "
                        f"{len(self._drug_index)} drugs, {len(self._disease_names)} diseases")
        except Exception as e:
            logger.warning(f"Index loading failed: {e}")

    def _query(self, cypher: str, params: dict = None):
        if not self.driver:
            return []
        records, _, _ = self.driver.execute_query(cypher, params or {})
        return records

    def build_from_json(self, json_path: str, batch_size: int = 200):
        """Import medical1.json into Neo4j using batch operations for speed."""
        logger.info(f"Importing data into Neo4j from {json_path}...")

        # Create constraints
        self._query("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Disease) REQUIRE d.name IS UNIQUE")
        self._query("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Symptom) REQUIRE s.name IS UNIQUE")
        self._query("CREATE CONSTRAINT IF NOT EXISTS FOR (dr:Drug) REQUIRE dr.name IS UNIQUE")
        self._query("CREATE CONSTRAINT IF NOT EXISTS FOR (dp:Department) REQUIRE dp.name IS UNIQUE")

        # Load all data into memory
        diseases = []
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                name = d.get("name", "")
                if not name:
                    continue
                diseases.append(d)

        total = len(diseases)

        for batch_start in range(0, total, batch_size):
            batch = diseases[batch_start:batch_start + batch_size]
            disease_names = []
            symptom_records = []
            drug_records = []
            dept_records = []
            complication_records = []

            for d in batch:
                name = d.get("name", "")
                disease_names.append({
                    "name": name, "desc": d.get("desc", ""),
                    "cause": d.get("cause", ""), "prevent": d.get("prevent", ""),
                    "cured_prob": d.get("cured_prob", ""),
                    "cost_money": d.get("cost_money", ""),
                })
                for s in (d.get("symptom") or []):
                    s = s.strip()
                    if s:
                        symptom_records.append({"d": name, "s": s})
                for rd in (d.get("recommand_drug") or []) + (d.get("common_drug") or []):
                    rd = rd.strip()
                    if rd:
                        drug_records.append({"d": name, "rd": rd})
                for dept in (d.get("cure_department") or []):
                    dept = dept.strip()
                    if dept:
                        dept_records.append({"d": name, "dept": dept})
                for comp in (d.get("acompany") or []):
                    comp = comp.strip()
                    if comp:
                        complication_records.append({"d": name, "comp": comp})

            # Batch create diseases
            self._query(
                "UNWIND $rows AS r "
                "MERGE (d:Disease {name: r.name}) "
                "SET d.desc = r.desc, d.cause = r.cause, d.prevent = r.prevent, "
                "d.cured_prob = r.cured_prob, d.cost_money = r.cost_money",
                {"rows": disease_names})

            # Batch create symptoms + relationships
            if symptom_records:
                self._query(
                    "UNWIND $rows AS r "
                    "MERGE (s:Symptom {name: r.s}) "
                    "WITH s, r MATCH (d:Disease {name: r.d}) "
                    "MERGE (d)-[:HAS_SYMPTOM]->(s)",
                    {"rows": symptom_records})

            # Batch create drugs + relationships
            if drug_records:
                self._query(
                    "UNWIND $rows AS r "
                    "MERGE (dd:Drug {name: r.rd}) "
                    "WITH dd, r MATCH (d:Disease {name: r.d}) "
                    "MERGE (d)-[:TREATS]->(dd)",
                    {"rows": drug_records})

            # Batch create departments + relationships
            if dept_records:
                self._query(
                    "UNWIND $rows AS r "
                    "MERGE (dp:Department {name: r.dept}) "
                    "WITH dp, r MATCH (d:Disease {name: r.d}) "
                    "MERGE (d)-[:TREATED_IN]->(dp)",
                    {"rows": dept_records})

            # Batch create complications
            if complication_records:
                self._query(
                    "UNWIND $rows AS r "
                    "MERGE (c:Disease {name: r.comp}) "
                    "WITH c, r MATCH (d:Disease {name: r.d}) "
                    "MERGE (d)-[:HAS_COMPLICATION]->(c)",
                    {"rows": complication_records})

            progress = min(batch_start + batch_size, total)
            logger.info(f"Imported {progress}/{total} diseases...")

        # Verify
        records = self._query("MATCH (d:Disease) RETURN COUNT(d) AS cnt")
        node_count = records[0]["cnt"] if records else 0
        logger.info(f"Neo4j import complete: {node_count} diseases")

    # ---- Query methods ----

    def get_disease_info(self, disease_name: str) -> dict | None:
        records = self._query(
            "MATCH (d:Disease {name: $name}) RETURN d", {"name": disease_name})
        if not records:
            return None
        info = dict(records[0]["d"])
        info["symptoms"] = self.get_symptoms_for_disease(disease_name)
        info["drugs"] = self.get_drugs_for_disease(disease_name)
        return info

    def get_diseases_by_symptoms(self, symptoms: list[str], top_k: int = 10) -> list[tuple[str, int]]:
        results = []
        for symptom in symptoms:
            records = self._query(
                "MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom) "
                "WHERE s.name CONTAINS $symptom OR $symptom CONTAINS s.name "
                "RETURN d.name AS name, COUNT(s) AS cnt "
                "ORDER BY cnt DESC LIMIT $n",
                {"symptom": symptom, "n": top_k})
            for r in records:
                results.append((r["name"], r["cnt"]))
        return results[:top_k]

    def get_drugs_for_disease(self, disease_name: str) -> list[str]:
        records = self._query(
            "MATCH (d:Disease {name: $name})-[:TREATS]->(dd:Drug) RETURN dd.name AS name",
            {"name": disease_name})
        return [r["name"] for r in records]

    def get_symptoms_for_disease(self, disease_name: str) -> list[str]:
        records = self._query(
            "MATCH (d:Disease {name: $name})-[:HAS_SYMPTOM]->(s:Symptom) RETURN s.name AS name",
            {"name": disease_name})
        return [r["name"] for r in records]

    def get_diseases_by_department(self, department: str) -> list[str]:
        records = self._query(
            "MATCH (d:Disease)-[:TREATED_IN]->(dp:Department {name: $dept}) "
            "RETURN d.name AS name ORDER BY name",
            {"dept": department})
        return [r["name"] for r in records]

    def search_diseases(self, keyword: str, top_k: int = 10) -> list[str]:
        records = self._query(
            "MATCH (d:Disease) WHERE d.name CONTAINS $kw "
            "RETURN d.name AS name LIMIT $n",
            {"kw": keyword, "n": top_k})
        return [r["name"] for r in records]

    def visualize_subgraph(self, disease_names: list[str], max_nodes: int = 60) -> tuple[str | None, int]:
        symptom_map = {}
        drug_map = {}
        comp_map = {}
        for d in disease_names:
            symptom_map[d] = self.get_symptoms_for_disease(d)
            drug_map[d] = self.get_drugs_for_disease(d)
            records = self._query(
                "MATCH (d:Disease {name: $name})-[:HAS_COMPLICATION]->(c:Disease) "
                "RETURN c.name AS name", {"name": d})
            comp_map[d] = [r["name"] for r in records]
        return render_subgraph_image(disease_names, symptom_map, drug_map, comp_map, max_nodes)

    def get_disease_count(self) -> int:
        records = self._query("MATCH (d:Disease) RETURN COUNT(d) AS cnt")
        return records[0]["cnt"] if records else 0


# ========================
# Factory function
# ========================


_kg_instance = None


def get_knowledge_graph(backend: str = "auto",
                        medical_json_path: str = "huatuo_data/medical1.json",
                        pickle_path: str = "data/knowledge_graph.pkl") -> KnowledgeGraphBackend:
    global _kg_instance
    if _kg_instance is not None:
        return _kg_instance

    # Try Neo4j first (unless explicitly told not to)
    if backend in ("auto", "neo4j"):
        try:
            kg = Neo4jBackend()
            records = kg._query("MATCH (d:Disease) RETURN COUNT(d) AS cnt")
            count = records[0]["cnt"] if records else 0
            if count == 0 and os.path.exists(medical_json_path):
                logger.info("Neo4j is empty, importing data...")
                kg.build_from_json(medical_json_path)
            _kg_instance = kg
            logger.info(f"Using Neo4j backend ({kg.get_disease_count()} diseases)")
            return _kg_instance
        except Exception as e:
            logger.warning(f"Neo4j unavailable: {e}")

    # Fallback to NetworkX
    if pickle_path and os.path.exists(pickle_path):
        _kg_instance = NetworkXBackend(pickle_path=pickle_path)
    elif medical_json_path and os.path.exists(medical_json_path):
        _kg_instance = NetworkXBackend(medical_json_path=medical_json_path)
        if pickle_path:
            _kg_instance.save_pickle(pickle_path)
    else:
        _kg_instance = NetworkXBackend()

    logger.info(f"Using NetworkX backend ({_kg_instance.get_disease_count()} diseases)")
    return _kg_instance
