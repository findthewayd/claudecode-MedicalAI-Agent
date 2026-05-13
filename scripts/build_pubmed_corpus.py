"""Generate pubmed_articles.txt from medical1.json"""
import json
import os

MEDICAL_JSON = "huatuo_data/medical1.json"
OUTPUT_FILE = "data/pubmed_articles.txt"


def build_corpus():
    if not os.path.exists(MEDICAL_JSON):
        print(f"Source file {MEDICAL_JSON} not found")
        return

    blocks = []
    with open(MEDICAL_JSON, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                disease = json.loads(line)
            except json.JSONDecodeError:
                continue

            parts = [f"疾病：{disease.get('name', '')}"]

            if disease.get("desc"):
                parts.append(f"描述：{disease['desc']}")

            symptoms = disease.get("symptom", [])
            if symptoms:
                parts.append(f"症状：{'、'.join(symptoms)}")

            cause = disease.get("cause", "")
            if cause:
                parts.append(f"病因：{cause}")

            cure_ways = disease.get("cure_way", [])
            if cure_ways:
                parts.append(f"治疗：{'、'.join(cure_ways)}")

            checks = disease.get("check", [])
            if checks:
                parts.append(f"检查：{'、'.join(checks)}")

            complications = disease.get("acompany", [])
            if complications:
                parts.append(f"并发症：{'、'.join(complications)}")

            drugs = disease.get("recommand_drug", []) or disease.get("common_drug", [])
            if drugs:
                parts.append(f"推荐药物：{'、'.join(drugs)}")

            blocks.append("\n".join(parts))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(blocks))

    print(f"Generated {OUTPUT_FILE} with {len(blocks)} disease entries")


if __name__ == "__main__":
    build_corpus()
