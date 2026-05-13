"""Build and pickle the NetworkX knowledge graph from medical1.json."""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_graph import NetworkXBackend

MEDICAL_JSON = "huatuo_data/medical1.json"
PICKLE_PATH = "data/knowledge_graph.pkl"


def main():
    print("Building knowledge graph from medical1.json...")
    start = time.time()

    kg = NetworkXBackend(medical_json_path=MEDICAL_JSON)
    kg.save_pickle(PICKLE_PATH)

    elapsed = time.time() - start
    print(f"\nGraph built in {elapsed:.1f}s")
    print(f"Diseases: {kg.get_disease_count()}")
    print(f"Total nodes: {kg.graph.number_of_nodes()}")
    print(f"Total edges: {kg.graph.number_of_edges()}")

    # Quick sanity check
    print("\n--- Sanity check ---")
    # Test disease lookup
    info = kg.get_disease_info("百日咳")
    if info:
        print(f"Disease: 百日咳")
        print(f"  Symptoms: {info.get('symptoms', [])[:5]}...")
        print(f"  Drugs: {info.get('drugs', [])[:5]}...")
        print(f"  Department: {info.get('departments', [])}")

    # Test symptom-based diagnosis
    results = kg.get_diseases_by_symptoms(["咳嗽", "发热", "乏力"])
    print(f"\nDifferential diagnosis for [咳嗽, 发热, 乏力]:")
    for disease, score in results[:10]:
        print(f"  {disease} (score: {score})")

    # Test department search
    dept_diseases = kg.get_diseases_by_department("呼吸内科")
    print(f"\nDiseases in 呼吸内科: {len(dept_diseases)}")

    # Test keyword search
    kw_results = kg.search_diseases("肺炎")
    print(f"\nSearch '肺炎': {kw_results[:10]}")

    print("\nDone!")


if __name__ == "__main__":
    main()
