import argparse
import json
import os
import re
import sys
import subprocess
from collections import defaultdict
from tqdm import tqdm

# --- 1. Import Evaluation Tools ---
try:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    import bert_score
    import torch
except ImportError as e:
    print(f"\n[ERROR] Missing dependency: {e}")
    print("Please install: pip install pycocoevalcap bert-score torch\n")
    sys.exit(1)

# --- 2. Evaluator Class ---
class TextEvaluator:
    def __init__(self):
        self.coco_scorers = []
        
        # 1. Add standard metrics
        self.coco_scorers.append((Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]))
        self.coco_scorers.append((Rouge(), "ROUGE_L"))
        self.coco_scorers.append((Cider(), "CIDEr"))
        
        # 2. Safely check for Java for METEOR
        self.has_meteor = False
        try:
            # Check if java exists in system path
            subprocess.run(["java", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            self.coco_scorers.append((Meteor(), "METEOR"))
            self.has_meteor = True
        except Exception:
            print("\n[WARNING] Java (JRE) not found or METEOR failed to init. Skipping METEOR.")
            print("To enable METEOR, run: sudo apt-get install default-jre\n")

    def evaluate(self, predictions, references):
        score_dict = {}
        if not predictions:
            return {}

        # Compute COCO Metrics
        for scorer, method in self.coco_scorers:
            try:
                score, _ = scorer.compute_score(references, predictions)
                if isinstance(method, list):
                    for sc, m in zip(score, method):
                        score_dict[m] = sc
                else:
                    score_dict[method] = score
            except Exception:
                if isinstance(method, list):
                    for m in method: score_dict[m] = 0.0
                else: score_dict[method] = 0.0

        # Compute BERTScore
        uids = sorted(predictions.keys())
        preds_list = [predictions[uid][0] for uid in uids]
        refs_list = [references[uid][0] for uid in uids]

        try:
            # Automatically uses GPU if available
            P, R, F1 = bert_score.score(preds_list, refs_list, lang="en", verbose=False)
            score_dict["BERTScore_F1"] = F1.mean().item()
        except Exception as e:
            print(f"BERTScore error: {e}")
            score_dict["BERTScore_F1"] = 0.0

        return score_dict

# --- 3. Helper Functions ---

def normalize_text(text):
    if not text: return ""
    text = text.lower().strip()
    # Keep punctuation and signs for coordinates
    return re.sub(r'[^\w\s\.\,\-\+]', '', text)

def load_and_match_data(pred_dir, gt_dir):
    results = defaultdict(lambda: {'preds': {}, 'gts': {}})
    
    # Get all prediction files (files without extensions)
    pred_filenames = [f for f in os.listdir(pred_dir) 
                     if not f.startswith('.') and os.path.isfile(os.path.join(pred_dir, f))]
    
    print(f"Found {len(pred_filenames)} files in prediction directory.")
    matched_count = 0

    for filename in tqdm(pred_filenames, desc="Processing Files"):
        pred_path = os.path.join(pred_dir, filename)
        token = filename 
        gt_path = os.path.join(gt_dir, token + ".json")
        
        if not os.path.exists(gt_path):
            continue
            
        try:
            with open(gt_path, 'r', encoding='utf-8') as f:
                gt_list = json.load(f)
            
            gt_lookup = {}
            for idx, item in enumerate(gt_list):
                q = item.get("question", "")
                a = item.get("answer", "")
                cat = "Scene Analysis" if idx == 0 else "Explanation" if idx == 1 else "Counterfactual"
                gt_lookup[normalize_text(q)] = {'answer': a, 'category': cat}

            with open(pred_path, 'r', encoding='utf-8') as f:
                pred_list = json.load(f)
            
            for p_item in pred_list:
                pred_q_raw = p_item.get("Q", "")
                pred_a_data = p_item.get("A", "")
                pred_a_str = pred_a_data[0] if isinstance(pred_a_data, list) else str(pred_a_data)

                q_norm = normalize_text(pred_q_raw)
                
                if q_norm in gt_lookup:
                    entry = gt_lookup[q_norm]
                    uid = f"{token}_{hash(q_norm)}"
                    cat = entry['category']
                    
                    for target in [cat, 'Overall']:
                        results[target]['preds'][uid] = [pred_a_str]
                        results[target]['gts'][uid]   = [entry['answer']]
                    matched_count += 1
        except Exception:
            pass
        
    return results, matched_count

# --- 4. Main Execution ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, required=True, help='Folder with extension-less prediction files')
    parser.add_argument('--gt_dir', type=str, required=True, help='Folder with .json ground truth files')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    data_map, total_samples = load_and_match_data(args.pred_dir, args.gt_dir)
    
    if total_samples == 0:
        print("\n[ERROR] No matching QA pairs found. Check if prediction filenames match GT filenames.")
        return

    print(f"\nSuccessfully matched {total_samples} QA pairs.")
    evaluator = TextEvaluator()

    # Define Column headers
    cols = ["CATEGORY", "BLEU-4"]
    if evaluator.has_meteor: cols.append("METEOR")
    cols += ["ROUGE-L", "CIDEr", "BERTScore"]
    
    # Calculate spacing
    header = f"{cols[0]:<20} | " + " | ".join([f"{c:<8}" for c in cols[1:]])
    print("\n" + "="*len(header))
    print(header)
    print("-" * len(header))

    desired_order = ["Scene Analysis", "Explanation", "Counterfactual", "Overall"]
    existing_cats = list(data_map.keys())
    sorted_cats = [c for c in desired_order if c in existing_cats]

    final_metrics = {}
    for cat in sorted_cats:
        scores = evaluator.evaluate(data_map[cat]['preds'], data_map[cat]['gts'])
        final_metrics[cat] = scores
        
        row = f"{cat:<20} | {scores.get('Bleu_4', 0):.4f}   | "
        if evaluator.has_meteor: row += f"{scores.get('METEOR', 0):.4f}   | "
        row += f"{scores.get('ROUGE_L', 0):.4f}   | {scores.get('CIDEr', 0):.4f}   | {scores.get('BERTScore_F1', 0):.4f}"
        print(row)

    print("="*len(header) + "\n")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(final_metrics, f, indent=4)

if __name__ == "__main__":
    main()