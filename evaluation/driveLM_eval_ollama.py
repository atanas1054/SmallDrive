# LLM-as-judge language score: an LLM served by Ollama rates each predicted answer
# against the ground truth on a 0-100 scale (reported as 0-1 in the paper).
import os
import re
import json
import argparse
import requests
from multiprocessing.pool import ThreadPool
from tqdm import tqdm 

class OllamaEvaluation:
    def __init__(self, base_url, model_name):
        if not base_url.endswith('/api/generate'):
            self.api_url = base_url.rstrip('/') + '/api/generate'
        else:
            self.api_url = base_url
        self.model_name = model_name

    def call_ollama(self, prompt):
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.6,      
                "seed": 42,            
                "num_predict": 40     
            }
        }
        try:
            response = requests.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json().get('response', '').strip()
        except Exception as e:
            return f"OLLAMA_ERROR: {e}"
    
    def forward(self, data):
        scene_id, q_name, answer, GT = data
        prompt = (
            "System: an evaluator who rates my answer based on the correct answer\n"
            "Rate my answer based on the correct answer out of 100, with higher scores indicating that the answer is closer to the correct answer, and you should be accurate to single digits like 62, 78, 41,etc. Output the number only.\n"
            f"Correct Answer: {GT}\n"
            f"My Answer: {answer}\n\n"
            "Score:"
        )
        reply = self.call_ollama(prompt)
        if "OLLAMA_ERROR" in reply:
            return 0.0

        number_match = re.search(r'\d+', reply)
        score = float(number_match.group(0)) if number_match else 0.0
        if score > 100: score = 100.0
        return score

def main():
    parser = argparse.ArgumentParser(description='Deterministic Ollama Evaluation')
    parser.add_argument('--pred_dir', type=str, required=True)
    parser.add_argument('--actions_gt_dir', type=str, required=True)
    parser.add_argument('--base_url', type=str, default="http://localhost:11434", help='Ollama server URL')
    parser.add_argument('--model', type=str, default="gemma3:27b", help='Judge model (the paper uses gemma3:27b)')
    parser.add_argument('--limit', type=int, default=None, help='Only evaluate the first N samples (default: all)')
    parser.add_argument('--workers', type=int, default=1) 
    args = parser.parse_args()

    eval_engine = OllamaEvaluation(args.base_url, args.model)
    all_evaluation_tasks = []

    gt_files = sorted([f for f in os.listdir(args.actions_gt_dir) if f.endswith('.json')])
    test_files = gt_files[:args.limit]

    for filename in test_files:
        scene_id = os.path.splitext(filename)[0]
        gt_path = os.path.join(args.actions_gt_dir, filename)
        pred_path = os.path.join(args.pred_dir, scene_id)
        if not os.path.exists(pred_path) and os.path.exists(pred_path + ".json"):
            pred_path += ".json"
        
        if not os.path.exists(pred_path): continue

        try:
            with open(gt_path, 'r') as f: gt_data = json.load(f)
            with open(pred_path, 'r') as f: pred_data = json.load(f)
            targets = [(1, 0, "Traffic"), (3, 2, "Traj 1"), (4, 3, "Traj 2")]
            for p_idx, g_idx, q_label in targets:
                if p_idx < len(pred_data) and g_idx < len(gt_data):
                    ans_raw = pred_data[p_idx].get('A', "")
                    answer = ans_raw[0] if isinstance(ans_raw, list) and len(ans_raw) > 0 else ans_raw
                    gt_ans = gt_data[g_idx].get('answer', "")
                    if answer and gt_ans:
                        all_evaluation_tasks.append((scene_id, q_label, answer, gt_ans))
        except: continue

    print(f"Running evaluation on {len(all_evaluation_tasks)} tasks...")

    # --- Step 2: Use tqdm with ThreadPool ---
    with ThreadPool(args.workers) as p:
        # p.imap allows us to iterate through results as they complete
        scores = list(tqdm(
            p.imap(eval_engine.forward, all_evaluation_tasks), 
            total=len(all_evaluation_tasks),
            desc="Evaluating"
        ))

    if scores:
        print(f"\nFINAL STABLE AVERAGE: {sum(scores) / len(scores):.2f}")

if __name__ == "__main__":
    main()