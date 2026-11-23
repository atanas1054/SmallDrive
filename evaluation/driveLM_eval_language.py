import argparse
import json
import os
import sys
import language_evaluation

# BLEU / ROUGE-L / CIDEr for scene description + next-action explanation.
# Requires: pip install git+https://github.com/bckim92/language-evaluation.git
class LanguageEvaluationSuit:
    def __init__(self):
        # Initialize only the language evaluator
        self.language_eval = language_evaluation.CocoEvaluator(coco_types=["BLEU", "ROUGE_L", "CIDEr"])
        self.language = {"answer": [], "GT": []}

    def add_pair(self, answer, gt):
        if answer and gt:
            self.language["answer"].append(str(answer))
            self.language["GT"].append(str(gt))

    def eval_language(self):
        answer = self.language["answer"]
        gt = self.language["GT"]
        if not answer:
            print("Warning: No data collected for evaluation.")
            return {}
        
        results_gen = self.language_eval.run_evaluation(answer, gt)
        return results_gen

def main():
    parser = argparse.ArgumentParser(description='Language Evaluation for Scene and Action')
    parser.add_argument('--pred_dir', type=str, required=True, help='Path to your prediction files')
    parser.add_argument('--scene_gt_dir', type=str, required=True, help='Path to ground truth (description)')
    parser.add_argument('--action_gt_dir', type=str, required=True, help='Path to ground truth (actions)')
    args = parser.parse_args()

    evaluator = LanguageEvaluationSuit()

    # We assume scene_gt_dir contains the master list of scene IDs
    gt_files = [f for f in os.listdir(args.scene_gt_dir) if f.endswith('.json')]
    
    cnt = 0
    for filename in gt_files:
        scene_id = os.path.splitext(filename)[0]
        
        # Construct Paths
        scene_gt_path = os.path.join(args.scene_gt_dir, filename)
        action_gt_path = os.path.join(args.action_gt_dir, filename)
        pred_path = os.path.join(args.pred_dir, scene_id)
        if not os.path.exists(pred_path) and os.path.exists(pred_path + ".json"):
            pred_path += ".json"

        if not (os.path.exists(pred_path) and os.path.exists(action_gt_path)):
            continue

        try:
            with open(pred_path, 'r') as f:
                pred_data = json.load(f)
            with open(scene_gt_path, 'r') as f:
                scene_gt_data = json.load(f)
            with open(action_gt_path, 'r') as f:
                action_gt_data = json.load(f)

            # --- 1. Extract Scene Description (First Prediction Q vs Scene GT) ---
            # Pred index 0, Q: "Can you summarize..."
            pred_desc = pred_data[0]['A'][0] if isinstance(pred_data[0]['A'], list) else pred_data[0]['A']
            gt_desc = scene_gt_data.get('description', '')
            evaluator.add_pair(pred_desc, gt_desc)

            # --- 2. Extract Next Action (Third Prediction Q vs Action GT Second Q) ---
            # Pred index 2, Q: "What should be your next action..."
            pred_action = pred_data[2]['A'][0] if isinstance(pred_data[2]['A'], list) else pred_data[2]['A']
            # Action GT index 1, Q: "What should be your next action..."
            gt_action = action_gt_data[1].get('answer', '')
            evaluator.add_pair(pred_action, gt_action)

        except (IndexError, KeyError, Exception) as e:
            print(f"Error processing {scene_id}: {e}")
            continue

        cnt += 1
        if cnt % 50 == 0:
            print(f"Processed {cnt} scenes...")

    print("\n--- Final Language Evaluation Results ---")
    results = evaluator.eval_language()
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")

if __name__ == '__main__':
    main()