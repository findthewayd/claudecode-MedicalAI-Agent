"""Generate drug_interactions.json with severity levels."""
import json, re, os

MEDICAL_JSON = "huatuo_data/medical1.json"
OUTPUT_FILE = "data/drug_interactions.json"

# Severity: 禁用=contraindicated, 慎用=caution, 注意=note
S_CONTRA = "禁用"
S_CAUTN = "慎用"
S_NOTE  = "注意"

# Expanded interaction rules with severity levels
INTERACTION_RULES = {
    # ===== Anticoagulants =====
    "华法林": [
        ("阿司匹林", S_CONTRA), ("布洛芬", S_CONTRA), ("萘普生", S_CONTRA),
        ("双氯芬酸", S_CONTRA), ("塞来昔布", S_CAUTN), ("对乙酰氨基酚", S_NOTE),
        ("奥美拉唑", S_CAUTN), ("西咪替丁", S_CAUTN), ("甲硝唑", S_CAUTN),
        ("氟康唑", S_CAUTN), ("胺碘酮", S_CONTRA), ("辛伐他汀", S_CAUTN),
        ("头孢类", S_CAUTN), ("左氧氟沙星", S_CAUTN), ("环丙沙星", S_CAUTN),
        ("克拉霉素", S_CAUTN), ("阿奇霉素", S_NOTE), ("四环素", S_NOTE),
        ("苯巴比妥", S_CAUTN), ("卡马西平", S_CAUTN), ("利福平", S_CAUTN),
        ("氯吡格雷", S_CONTRA),
    ],
    "肝素": [
        ("阿司匹林", S_CONTRA), ("布洛芬", S_CONTRA), ("华法林", S_CONTRA),
        ("氯吡格雷", S_CONTRA), ("双嘧达莫", S_CONTRA),
    ],
    "氯吡格雷": [
        ("奥美拉唑", S_CONTRA), ("埃索美拉唑", S_CONTRA), ("华法林", S_CONTRA),
        ("阿司匹林", S_CONTRA), ("肝素", S_CONTRA), ("布洛芬", S_CONTRA),
        ("西洛他唑", S_CONTRA),
    ],
    "利伐沙班": [
        ("阿司匹林", S_CONTRA), ("布洛芬", S_CONTRA), ("华法林", S_CONTRA),
        ("氯吡格雷", S_CONTRA), ("酮康唑", S_CONTRA), ("伊曲康唑", S_CONTRA),
        ("利福平", S_CAUTN), ("卡马西平", S_CAUTN),
    ],

    # ===== NSAIDs =====
    "阿司匹林": [
        ("布洛芬", S_CONTRA), ("萘普生", S_CONTRA), ("双氯芬酸", S_CONTRA),
        ("华法林", S_CONTRA), ("肝素", S_CONTRA), ("甲氨蝶呤", S_CONTRA),
        ("螺内酯", S_CAUTN), ("呋塞米", S_CAUTN), ("地高辛", S_CAUTN),
        ("胰岛素", S_CAUTN), ("格列本脲", S_CAUTN), ("丙磺舒", S_CAUTN),
        ("泼尼松", S_CONTRA), ("氢氯噻嗪", S_NOTE),
    ],
    "布洛芬": [
        ("阿司匹林", S_CONTRA), ("华法林", S_CONTRA), ("甲氨蝶呤", S_CONTRA),
        ("呋塞米", S_CAUTN), ("氢氯噻嗪", S_CAUTN), ("地高辛", S_CAUTN),
        ("环孢素", S_CAUTN), ("他克莫司", S_CAUTN), ("锂剂", S_CONTRA),
        ("赖诺普利", S_CAUTN), ("卡托普利", S_CAUTN), ("缬沙坦", S_CAUTN),
    ],
    "双氯芬酸": [
        ("阿司匹林", S_CONTRA), ("华法林", S_CONTRA), ("甲氨蝶呤", S_CONTRA),
        ("环孢素", S_CAUTN), ("地高辛", S_CAUTN), ("呋塞米", S_CAUTN),
        ("氢氯噻嗪", S_CAUTN), ("锂剂", S_CONTRA), ("酮咯酸", S_CONTRA),
    ],
    "对乙酰氨基酚": [
        ("华法林", S_NOTE), ("酒精", S_CONTRA), ("卡马西平", S_CAUTN),
        ("异烟肼", S_CONTRA), ("苯巴比妥", S_CAUTN),
    ],
    "塞来昔布": [
        ("华法林", S_CAUTN), ("阿司匹林", S_CONTRA), ("氟康唑", S_CAUTN),
        ("锂剂", S_CAUTN), ("呋塞米", S_CAUTN), ("卡托普利", S_CAUTN),
    ],

    # ===== Antibiotics =====
    "阿莫西林": [
        ("甲氨蝶呤", S_CONTRA), ("华法林", S_CAUTN), ("四环素", S_NOTE),
        ("氯霉素", S_NOTE), ("别嘌醇", S_CAUTN),
    ],
    "头孢地尼": [
        ("呋塞米", S_CAUTN), ("华法林", S_CAUTN), ("庆大霉素", S_CAUTN),
    ],
    "头孢曲松": [
        ("钙剂", S_CONTRA), ("呋塞米", S_CAUTN), ("庆大霉素", S_CAUTN),
        ("华法林", S_CAUTN),
    ],
    "左氧氟沙星": [
        ("华法林", S_CAUTN), ("茶碱", S_CAUTN), ("环孢素", S_CAUTN),
        ("铁剂", S_NOTE), ("钙剂", S_NOTE), ("氢氧化铝", S_NOTE),
        ("格列本脲", S_CAUTN), ("泼尼松", S_CAUTN),
    ],
    "莫西沙星": [
        ("华法林", S_CAUTN), ("胺碘酮", S_CONTRA), ("索他洛尔", S_CONTRA),
        ("西沙必利", S_CONTRA), ("茶碱", S_CAUTN),
    ],
    "甲硝唑": [
        ("华法林", S_CAUTN), ("酒精", S_CONTRA), ("苯妥英钠", S_CAUTN),
        ("锂剂", S_CAUTN), ("环孢素", S_CAUTN), ("氟尿嘧啶", S_CAUTN),
        ("西咪替丁", S_NOTE),
    ],
    "克拉霉素": [
        ("华法林", S_CAUTN), ("地高辛", S_CAUTN), ("辛伐他汀", S_CONTRA),
        ("阿托伐他汀", S_CONTRA), ("卡马西平", S_CAUTN), ("茶碱", S_CAUTN),
        ("秋水仙碱", S_CONTRA), ("麦角胺", S_CONTRA),
    ],
    "阿奇霉素": [
        ("华法林", S_CAUTN), ("地高辛", S_CAUTN), ("环孢素", S_CAUTN),
        ("麦角胺", S_CONTRA),
    ],
    "四环素": [
        ("阿莫西林", S_NOTE), ("铁剂", S_NOTE), ("钙剂", S_NOTE),
        ("氢氧化铝", S_NOTE), ("华法林", S_CAUTN), ("维A酸", S_CONTRA),
        ("甲氧氟烷", S_CONTRA),
    ],
    "庆大霉素": [
        ("呋塞米", S_CONTRA), ("头孢地尼", S_CAUTN), ("万古霉素", S_CAUTN),
        ("顺铂", S_CONTRA), ("环孢素", S_CAUTN), ("两性霉素B", S_CONTRA),
    ],

    # ===== ACE Inhibitors / ARBs =====
    "卡托普利": [
        ("螺内酯", S_CONTRA), ("氯化钾", S_CONTRA), ("锂剂", S_CONTRA),
        ("别嘌醇", S_CAUTN), ("布洛芬", S_CAUTN), ("阿司匹林", S_CAUTN),
        ("二甲双胍", S_NOTE),
    ],
    "依那普利": [
        ("螺内酯", S_CONTRA), ("氯化钾", S_CONTRA), ("锂剂", S_CONTRA),
        ("布洛芬", S_CAUTN),
    ],
    "缬沙坦": [
        ("螺内酯", S_CAUTN), ("氯化钾", S_CAUTN), ("锂剂", S_CAUTN),
        ("布洛芬", S_NOTE),
    ],
    "氯沙坦": [
        ("螺内酯", S_CAUTN), ("氯化钾", S_CAUTN), ("锂剂", S_CAUTN),
        ("氟康唑", S_CAUTN), ("利福平", S_CAUTN),
    ],

    # ===== Statins =====
    "辛伐他汀": [
        ("克拉霉素", S_CONTRA), ("伊曲康唑", S_CONTRA), ("环孢素", S_CONTRA),
        ("华法林", S_CAUTN), ("地高辛", S_CAUTN), ("吉非贝齐", S_CONTRA),
        ("酮康唑", S_CONTRA), ("胺碘酮", S_CAUTN), ("氟康唑", S_CAUTN),
        ("秋水仙碱", S_CAUTN),
    ],
    "阿托伐他汀": [
        ("克拉霉素", S_CAUTN), ("伊曲康唑", S_CAUTN), ("环孢素", S_CAUTN),
        ("华法林", S_CAUTN), ("地高辛", S_CAUTN), ("秋水仙碱", S_CAUTN),
        ("吉非贝齐", S_CAUTN),
    ],

    # ===== Diuretics =====
    "呋塞米": [
        ("阿司匹林", S_CAUTN), ("布洛芬", S_CAUTN), ("头孢地尼", S_CAUTN),
        ("地高辛", S_CAUTN), ("庆大霉素", S_CONTRA), ("锂剂", S_CONTRA),
        ("卡托普利", S_CAUTN),
    ],
    "氢氯噻嗪": [
        ("布洛芬", S_CAUTN), ("锂剂", S_CONTRA), ("地高辛", S_CAUTN),
        ("降糖药", S_CAUTN), ("二甲双胍", S_NOTE),
    ],
    "螺内酯": [
        ("氯化钾", S_CONTRA), ("卡托普利", S_CONTRA), ("依那普利", S_CONTRA),
        ("阿司匹林", S_CAUTN), ("缬沙坦", S_CAUTN), ("氯沙坦", S_CAUTN),
    ],

    # ===== Diabetes =====
    "二甲双胍": [
        ("酒精", S_CONTRA), ("碘造影剂", S_CONTRA), ("呋塞米", S_NOTE),
        ("卡托普利", S_NOTE), ("西咪替丁", S_CAUTN),
    ],
    "格列本脲": [
        ("阿司匹林", S_CAUTN), ("华法林", S_CAUTN), ("氯霉素", S_CAUTN),
        ("左氧氟沙星", S_CAUTN), ("保泰松", S_CONTRA),
    ],
    "胰岛素": [
        ("阿司匹林", S_CAUTN), ("氢氯噻嗪", S_CAUTN), ("普萘洛尔", S_CAUTN),
        ("泼尼松", S_CAUTN), ("酒精", S_CAUTN), ("噻唑烷二酮", S_CAUTN),
    ],

    # ===== CNS =====
    "地西泮": [
        ("酒精", S_CONTRA), ("苯巴比妥", S_CONTRA), ("西咪替丁", S_CAUTN),
        ("奥美拉唑", S_CAUTN), ("氯氮平", S_CONTRA), ("卡马西平", S_CAUTN),
        ("酮康唑", S_CONTRA),
    ],
    "苯巴比妥": [
        ("华法林", S_CAUTN), ("地西泮", S_CONTRA), ("苯妥英钠", S_CAUTN),
        ("环孢素", S_CAUTN), ("对乙酰氨基酚", S_CAUTN), ("丙戊酸", S_CAUTN),
        ("卡马西平", S_CAUTN), ("激素类避孕药", S_CONTRA),
    ],
    "卡马西平": [
        ("华法林", S_CAUTN), ("克拉霉素", S_CAUTN), ("环孢素", S_CAUTN),
        ("对乙酰氨基酚", S_CAUTN), ("丙戊酸", S_CAUTN), ("苯巴比妥", S_CAUTN),
        ("激素类避孕药", S_CONTRA), ("地西泮", S_CAUTN),
    ],
    "锂剂": [
        ("布洛芬", S_CONTRA), ("双氯芬酸", S_CONTRA), ("萘普生", S_CONTRA),
        ("氢氯噻嗪", S_CONTRA), ("呋塞米", S_CONTRA), ("卡托普利", S_CONTRA),
        ("甲硝唑", S_CAUTN), ("塞来昔布", S_CAUTN),
    ],
    "舍曲林": [
        ("单胺氧化酶抑制剂", S_CONTRA), ("华法林", S_CAUTN), ("阿司匹林", S_CAUTN),
        ("曲坦类药物", S_CONTRA), ("利奈唑胺", S_CONTRA),
    ],
    "氟西汀": [
        ("单胺氧化酶抑制剂", S_CONTRA), ("华法林", S_CAUTN), ("锂剂", S_CAUTN),
        ("曲坦类药物", S_CONTRA), ("硫利达嗪", S_CONTRA),
    ],
    "阿米替林": [
        ("单胺氧化酶抑制剂", S_CONTRA), ("肾上腺素", S_CONTRA),
        ("沙丁胺醇", S_CAUTN), ("可乐定", S_CAUTN),
    ],

    # ===== Antifungals =====
    "氟康唑": [
        ("华法林", S_CAUTN), ("苯妥英钠", S_CAUTN), ("环孢素", S_CAUTN),
        ("辛伐他汀", S_CAUTN), ("西沙必利", S_CONTRA), ("匹莫齐特", S_CONTRA),
        ("利福平", S_CAUTN),
    ],
    "酮康唑": [
        ("辛伐他汀", S_CONTRA), ("华法林", S_CONTRA), ("地西泮", S_CONTRA),
        ("西沙必利", S_CONTRA), ("匹莫齐特", S_CONTRA), ("利福平", S_CONTRA),
    ],
    "伊曲康唑": [
        ("辛伐他汀", S_CONTRA), ("阿托伐他汀", S_CAUTN), ("地西泮", S_CAUTN),
        ("华法林", S_CAUTN), ("西沙必利", S_CONTRA),
    ],

    # ===== Proton Pump Inhibitors =====
    "奥美拉唑": [
        ("氯吡格雷", S_CONTRA), ("华法林", S_CAUTN), ("地西泮", S_CAUTN),
        ("苯妥英钠", S_CAUTN), ("甲氨蝶呤", S_CAUTN), ("酮康唑", S_CAUTN),
    ],
    "埃索美拉唑": [
        ("氯吡格雷", S_CONTRA), ("华法林", S_CAUTN),
    ],

    # ===== Calcium Channel Blockers =====
    "硝苯地平": [
        ("利福平", S_CONTRA), ("克拉霉素", S_CAUTN), ("伊曲康唑", S_CAUTN),
        ("地高辛", S_CAUTN), ("苯妥英钠", S_CAUTN),
    ],
    "维拉帕米": [
        ("地高辛", S_CAUTN), ("β阻滞剂", S_CONTRA), ("辛伐他汀", S_CAUTN),
        ("卡马西平", S_CAUTN), ("利福平", S_CONTRA),
    ],
    "地尔硫卓": [
        ("辛伐他汀", S_CAUTN), ("地高辛", S_CAUTN), ("β阻滞剂", S_CAUTN),
        ("卡马西平", S_CAUTN),
    ],

    # ===== Others =====
    "地高辛": [
        ("呋塞米", S_CAUTN), ("氢氯噻嗪", S_CAUTN), ("阿莫西林", S_NOTE),
        ("克拉霉素", S_CAUTN), ("伊曲康唑", S_CAUTN), ("维拉帕米", S_CAUTN),
        ("硝苯地平", S_CAUTN), ("布洛芬", S_CAUTN), ("螺内酯", S_CAUTN),
        ("两性霉素B", S_CAUTN),
    ],
    "甲氨蝶呤": [
        ("阿司匹林", S_CONTRA), ("布洛芬", S_CONTRA), ("双氯芬酸", S_CONTRA),
        ("萘普生", S_CONTRA), ("奥美拉唑", S_CAUTN), ("阿莫西林", S_CONTRA),
        ("磺胺类", S_CONTRA), ("复方新诺明", S_CONTRA),
    ],
    "环孢素": [
        ("布洛芬", S_CAUTN), ("双氯芬酸", S_CAUTN), ("阿奇霉素", S_CAUTN),
        ("克拉霉素", S_CAUTN), ("氟康唑", S_CAUTN), ("辛伐他汀", S_CONTRA),
        ("左氧氟沙星", S_CAUTN), ("苯巴比妥", S_CAUTN), ("卡马西平", S_CAUTN),
        ("利福平", S_CONTRA), ("氯化钾", S_CAUTN),
    ],
    "他克莫司": [
        ("布洛芬", S_CAUTN), ("氟康唑", S_CAUTN), ("酮康唑", S_CONTRA),
        ("伊曲康唑", S_CONTRA), ("克拉霉素", S_CONTRA), ("环孢素", S_CONTRA),
        ("利福平", S_CAUTN),
    ],
    "秋水仙碱": [
        ("克拉霉素", S_CONTRA), ("辛伐他汀", S_CONTRA), ("环孢素", S_CONTRA),
        ("阿托伐他汀", S_CAUTN),
    ],
    "别嘌醇": [
        ("阿莫西林", S_CAUTN), ("卡托普利", S_CAUTN), ("华法林", S_CAUTN),
        ("氢氯噻嗪", S_CAUTN), ("硫唑嘌呤", S_CONTRA),
    ],
    "茶碱": [
        ("左氧氟沙星", S_CAUTN), ("环丙沙星", S_CAUTN), ("克拉霉素", S_CAUTN),
        ("卡马西平", S_CAUTN), ("苯巴比妥", S_CAUTN), ("西咪替丁", S_CAUTN),
    ],
    "苯妥英钠": [
        ("苯巴比妥", S_CAUTN), ("卡马西平", S_CAUTN), ("甲硝唑", S_CAUTN),
        ("氟康唑", S_CAUTN), ("奥美拉唑", S_CAUTN), ("华法林", S_CAUTN),
        ("硝苯地平", S_CAUTN),
    ],
    "丙戊酸": [
        ("苯巴比妥", S_CONTRA), ("卡马西平", S_CAUTN), ("拉莫三嗪", S_CAUTN),
        ("华法林", S_NOTE), ("阿司匹林", S_CAUTN),
    ],
    "胺碘酮": [
        ("华法林", S_CONTRA), ("辛伐他汀", S_CAUTN), ("地高辛", S_CAUTN),
        ("莫西沙星", S_CONTRA), ("左氧氟沙星", S_CONTRA), ("索他洛尔", S_CONTRA),
    ],
    "利福平": [
        ("华法林", S_CONTRA), ("硝苯地平", S_CONTRA), ("环孢素", S_CONTRA),
        ("酮康唑", S_CONTRA), ("激素类避孕药", S_CONTRA),
        ("利伐沙班", S_CAUTN), ("维拉帕米", S_CONTRA),
    ],
    "左甲状腺素": [
        ("铁剂", S_NOTE), ("钙剂", S_NOTE), ("氢氧化铝", S_NOTE),
        ("华法林", S_CAUTN), ("舍曲林", S_NOTE),
    ],
    "西咪替丁": [
        ("华法林", S_CAUTN), ("地西泮", S_CAUTN), ("茶碱", S_CAUTN),
        ("苯妥英钠", S_CAUTN), ("二甲双胍", S_CAUTN),
    ],
    "泼尼松": [
        ("阿司匹林", S_CONTRA), ("布洛芬", S_CONTRA), ("胰岛素", S_CAUTN),
        ("华法林", S_CAUTN), ("左氧氟沙星", S_CAUTN),
    ],
    "甲氧氯普胺": [
        ("左旋多巴", S_CONTRA), ("地高辛", S_NOTE), ("环孢素", S_CAUTN),
    ],
    "曲坦类": [
        ("舍曲林", S_CONTRA), ("氟西汀", S_CONTRA), ("单胺氧化酶抑制剂", S_CONTRA),
        ("麦角胺", S_CONTRA),
    ],
    "西沙必利": [
        ("酮康唑", S_CONTRA), ("伊曲康唑", S_CONTRA), ("氟康唑", S_CONTRA),
        ("克拉霉素", S_CONTRA), ("莫西沙星", S_CONTRA), ("胺碘酮", S_CONTRA),
    ],
}


def normalize_name(name):
    return name.replace(" ", "").replace("　", "")


FORMULATION_SUFFIXES = [
    "肠溶片", "缓释片", "控释片", "分散片", "泡腾片", "咀嚼片", "含片",
    "肠溶胶囊", "缓释胶囊", "软胶囊", "胶囊", "片",
    "颗粒", "糖浆", "口服液", "口服溶液", "口服混悬液",
    "注射液", "注射用", "注射剂", "粉针", "冻干粉针",
    "滴眼液", "滴耳液", "滴鼻液", "眼膏", "软膏", "乳膏", "凝胶",
    "栓", "贴剂", "贴片", "气雾剂", "喷雾剂", "吸入剂",
    "肠溶丸", "缓释微丸", "丸", "散", "膏",
]


def strip_formulation(name):
    for suffix in sorted(FORMULATION_SUFFIXES, key=len, reverse=True):
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[:-len(suffix)]
    return name


def extract_all_drugs():
    drug_diseases = {}
    with open(MEDICAL_JSON, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: d = json.loads(line)
            except json.JSONDecodeError: continue
            disease_name = d.get("name", "")
            raw_drugs = (d.get("recommand_drug", []) or []) + (d.get("common_drug", []) or [])
            for rd in raw_drugs:
                rd = rd.strip()
                if not rd: continue
                match = re.search(r"\(([^)]+)\)", rd)
                name = normalize_name(match.group(1) if match else rd)
                if name:
                    drug_diseases.setdefault(name, set()).add(disease_name)
    return drug_diseases


def fuzzy_match_drug(rule_name, drug_map):
    rule_norm = normalize_name(rule_name)
    if not rule_norm: return []
    matches = []
    for norm_name, (orig_name, stripped_name) in drug_map.items():
        if rule_norm in norm_name or rule_norm in stripped_name:
            matches.append(orig_name)
        elif norm_name in rule_norm or stripped_name in rule_norm:
            matches.append(orig_name)
    return matches


def build_interactions(drug_diseases):
    all_drugs = set(drug_diseases.keys())
    drug_map = {}
    for d in all_drugs:
        norm = normalize_name(d)
        drug_map[norm] = (d, strip_formulation(norm))

    result = {}
    for rule_drug, rule_list in INTERACTION_RULES.items():
        matched_drugs = fuzzy_match_drug(rule_drug, drug_map)
        if not matched_drugs:
            rule_norm = normalize_name(rule_drug)
            if rule_norm in drug_map:
                matched_drugs = [drug_map[rule_norm][0]]
        if not matched_drugs: continue

        for matched_drug in matched_drugs:
            result.setdefault(matched_drug, {})
            for interacting_name, severity in rule_list:
                matched_others = fuzzy_match_drug(interacting_name, drug_map)
                for mo in matched_others:
                    if mo != matched_drug:
                        if mo not in result[matched_drug] or severity_rank(severity) > severity_rank(result[matched_drug].get(mo, S_NOTE)):
                            result[matched_drug][mo] = severity
                        result.setdefault(mo, {})
                        if matched_drug not in result[mo] or severity_rank(severity) > severity_rank(result[mo].get(matched_drug, S_NOTE)):
                            result[mo][matched_drug] = severity
    return result


def severity_rank(s):
    return {S_CONTRA: 3, S_CAUTN: 2, S_NOTE: 1}.get(s, 0)


def main():
    print("Extracting drug data from medical1.json...")
    drug_diseases = extract_all_drugs()
    print(f"Found {len(drug_diseases)} unique drugs across diseases")

    print("Building interaction database with severity levels...")
    interactions = build_interactions(drug_diseases)
    print(f"Generated interactions for {len(interactions)} drugs")
    stats = {S_CONTRA: 0, S_CAUTN: 0, S_NOTE: 0}
    for drug, others in interactions.items():
        for sev in others.values():
            if sev in stats: stats[sev] += 1
    print(f"  禁用: {stats[S_CONTRA]} | 慎用: {stats[S_CAUTN]} | 注意: {stats[S_NOTE]}")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(interactions, f, ensure_ascii=False, indent=2)
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
