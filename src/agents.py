import json
import pandas as pd

class ECGPredictionSummaryAgent:
    def process(self, ecg_id, prob, pred_label, true_label=None):
        summary = {
            "ecg_id": ecg_id,
            "predicted_probability": float(prob),
            "predicted_label": int(pred_label)
        }
        if true_label is not None:
            summary["true_label"] = int(true_label)
            if pred_label == 1 and true_label == 1:
                summary["case_type"] = "True Positive"
            elif pred_label == 1 and true_label == 0:
                summary["case_type"] = "False Positive"
            elif pred_label == 0 and true_label == 1:
                summary["case_type"] = "False Negative"
            else:
                summary["case_type"] = "True Negative"
        else:
            summary["case_type"] = "Unknown"
            
        return summary

class RiskInterpretationAgent:
    def process(self, prob):
        if prob < 0.30:
            risk = "Low risk"
        elif 0.30 <= prob < 0.70:
            risk = "Intermediate risk"
        else:
            risk = "High risk"
            
        interpretation = (f"{risk} (Probability: {prob:.3f}). "
                          "Note: This is an ECG-model-based risk category, not a clinical diagnosis.")
        return risk, interpretation

class ExplanationRecommendationAgent:
    def process(self, summary, risk, interpretation):
        explanation = (
            f"The ECG-based model predicted a {risk.lower()} of myocardial infarction "
            f"for this recording (Probability = {summary['predicted_probability']:.3f}). "
        )
        if summary['case_type'] != "Unknown":
            explanation += f"In retrospective analysis, this case was a {summary['case_type']}. "
            
        explanation += (
            "This output is based only on ECG signal patterns learned from the PTB-XL dataset. "
            "Since no laboratory values, biomarkers (e.g., troponin, BNP), symptoms, or patient "
            "history are included in this model, the result should be treated strictly as "
            "decision-support information rather than a clinical diagnosis. "
            "Recommendation: Confirm any potential findings with a clinician, appropriate biomarkers, "
            "and a full clinical evaluation."
        )
        return explanation

class AgenticWorkflow:
    def __init__(self):
        self.summary_agent = ECGPredictionSummaryAgent()
        self.risk_agent = RiskInterpretationAgent()
        self.explanation_agent = ExplanationRecommendationAgent()
        
    def run(self, ecg_id, prob, pred_label, true_label=None):
        summary = self.summary_agent.process(ecg_id, prob, pred_label, true_label)
        risk, interpretation = self.risk_agent.process(prob)
        explanation = self.explanation_agent.process(summary, risk, interpretation)
        
        result = {
            "ecg_id": ecg_id,
            "true_label": summary.get("true_label", None),
            "predicted_probability": summary["predicted_probability"],
            "predicted_label": summary["predicted_label"],
            "case_type": summary["case_type"],
            "risk_category": risk,
            "explanation": explanation
        }
        return result

def run_error_analysis(predictions_path='results/test_predictions.csv'):
    print("Running Agentic Error Analysis...")
    df = pd.read_csv(predictions_path)
    
    workflow = AgenticWorkflow()
    
    # Categorize cases
    tp_df = df[(df.predicted_label == 1) & (df.true_label == 1)]
    fp_df = df[(df.predicted_label == 1) & (df.true_label == 0)]
    fn_df = df[(df.predicted_label == 0) & (df.true_label == 1)]
    tn_df = df[(df.predicted_label == 0) & (df.true_label == 0)]
    
    # Select up to 5 examples from each
    selected_cases = pd.concat([
        tp_df.head(5),
        fp_df.head(5),
        fn_df.head(5),
        tn_df.head(5)
    ])
    
    results = []
    for _, row in selected_cases.iterrows():
        res = workflow.run(
            ecg_id=row['ecg_id'] if 'ecg_id' in row else row.name, # handle index vs column
            prob=row['predicted_probability'],
            pred_label=row['predicted_label'],
            true_label=row['true_label']
        )
        results.append(res)
        
    out_df = pd.DataFrame(results)
    out_df.to_csv('results/error_analysis.csv', index=False)
    print("Error analysis saved to results/error_analysis.csv")
    
    with open('results/explanation_examples.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Explanation examples saved to results/explanation_examples.json")

if __name__ == "__main__":
    run_error_analysis()
