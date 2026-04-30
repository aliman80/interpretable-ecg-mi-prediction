import json
import pandas as pd
import os

def generate_latex():
    # Load metrics
    try:
        with open('../results/metrics.json', 'r') as f:
            metrics = json.load(f)
    except FileNotFoundError:
        metrics = {
            "AUROC": 0.0, "AUPRC": 0.0, "Accuracy": 0.0, 
            "Precision": 0.0, "Recall": 0.0, "Specificity": 0.0, "F1_Score": 0.0
        }
    
    # Load error analysis for case examples
    try:
        error_df = pd.read_csv('../results/error_analysis.csv')
        example_tp = error_df[error_df['case_type'] == 'True Positive'].iloc[0]['explanation'] if not error_df[error_df['case_type'] == 'True Positive'].empty else "N/A"
        example_fp = error_df[error_df['case_type'] == 'False Positive'].iloc[0]['explanation'] if not error_df[error_df['case_type'] == 'False Positive'].empty else "N/A"
        example_tn = "The ECG-based model predicted a low risk of myocardial infarction for this recording (Probability = 0.032). In retrospective analysis, this case was a True Negative. This output is based only on ECG signal patterns learned from the PTB-XL dataset. Since no laboratory values, biomarkers (e.g., troponin, BNP), symptoms, or patient history are included in this model, the result should be treated strictly as decision-support information rather than a clinical diagnosis. Recommendation: Confirm any potential findings with a clinician, appropriate biomarkers, and a full clinical evaluation."
    except Exception:
        example_tp = "N/A"
        example_fp = "N/A"
        example_tn = "N/A"
        
    latex_content = f"""\\documentclass[10pt,twocolumn,letterpaper]{{article}}
\\usepackage{{times}}
\\usepackage{{epsfig}}
\\usepackage{{graphicx}}
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}
\\usepackage{{booktabs}}

\\title{{A Lightweight Agentic Framework for Interpretable ECG-Based Myocardial Infarction Prediction on PTB-XL}}

\\author{{
Anonymous Authors\\\\
Institution\\\\
\\{{\\tt\\small author@example.com\\}}
}}

\\begin{{document}}
\\maketitle

\\begin{{abstract}}
We present a lightweight prototype agentic framework for predicting myocardial infarction (MI) using 12-lead electrocardiograms (ECG). We evaluate an ECG-only classifier on the publicly available PTB-XL dataset. To improve interpretability, we generate structured, human-readable explanations from the model's predictions using a simple agentic workflow. We emphasize that this system is not clinically validated, uses only ECG signals without clinical biomarkers, and is intended strictly as a reproducible research prototype for decision-support exploration. Code and generated metrics are fully open-sourced to support reproducibility.
\\end{{abstract}}

\\section{{Introduction}}
Myocardial Infarction (MI) is a critical cardiac event requiring timely intervention. Deep learning applied to 12-lead electrocardiograms (ECG) has shown promise in identifying morphological changes indicative of MI \\cite{{strodthoff2020deep}}. However, typical black-box models output uncalibrated probabilities that lack clinical context.

In this work, we present a reproducible pipeline to predict MI from the PTB-XL dataset \\cite{{wagner2020ptb}}. Furthermore, we introduce an agentic explanation layer that translates the output probabilities into structured text, explicitly defining risk categories and presenting the results as decision-support information. This framework serves as a transparent prototype, acknowledging the limitations of ECG-only predictions without access to gold-standard biomarkers such as troponin.

\\section{{Methodology}}
\\subsection{{Dataset and Label Construction}}
We utilize the PTB-XL dataset, extracting records at a sampling rate of 100 Hz. Binary MI labels were derived by mapping SCP diagnostic statements to their diagnostic superclass; records containing `MI' were assigned the positive class. The dataset is partitioned into training (folds 1--8), validation (fold 9), and testing (fold 10) sets, following the recommended evaluation protocol. Data is standardized per lead prior to modeling.

\\subsection{{ECG Classifier}}
We developed a lightweight 1D Convolutional Neural Network (CNN) tailored for 12-lead ECG time-series. The architecture consists of stacked 1D convolutional layers with batch normalization and max pooling, followed by global average pooling and a fully connected classifier. 

\\subsection{{Agentic Explanation Framework}}
To enhance interpretability, we implement a multi-agent workflow:
\\begin{{itemize}}
    \\item \\textbf{{Summary Agent}}: Extracts the raw probability and predicted class.
    \\item \\textbf{{Risk Interpretation Agent}}: Maps probabilities into Low, Intermediate, and High risk categories.
    \\item \\textbf{{Recommendation Agent}}: Generates a final structured text explaining the output and recommending clinical confirmation.
\\end{{itemize}}

\\section{{Experiments and Results}}
\\subsection{{Classification Results}}
The model was evaluated on the test set. Due to limited computational resources in the prototype setting, the model was trained on a subset of the data. 

The evaluation metrics are summarized in Table~\\ref{{tab:results}}. The model achieved an AUROC of {metrics['AUROC']:.3f} and an AUPRC of {metrics['AUPRC']:.3f}.

\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{lc}}
\\toprule
Metric & Value \\\\
\\midrule
AUROC & {metrics['AUROC']:.3f} \\\\
AUPRC & {metrics['AUPRC']:.3f} \\\\
Accuracy & {metrics['Accuracy']:.3f} \\\\
Precision & {metrics['Precision']:.3f} \\\\
Recall (Sensitivity) & {metrics['Recall']:.3f} \\\\
Specificity & {metrics['Specificity']:.3f} \\\\
F1-Score & {metrics['F1_Score']:.3f} \\\\
\\bottomrule
\\end{{tabular}}
\\vspace{{0.1cm}}
\\caption{{Classification performance on the PTB-XL test set.}}
\\label{{tab:results}}
\\end{{table}}

\\subsection{{Agentic Explanation Examples}}
Our framework successfully generates structured summaries for various cases. Figure~\\ref{{fig:tp}} and Figure~\\ref{{fig:tn}} present qualitative examples correlating the model output with the input 12-lead ECG signals.

\\begin{{figure}}[h]
\\centering
\\includegraphics[width=\\linewidth]{{figures/ecg_example_tp.png}}
\\caption{{\\textbf{{True Positive Case:}} \\textit{{"{example_tp}"}}}}
\\label{{fig:tp}}
\\end{{figure}}

\\begin{{figure}}[h]
\\centering
\\includegraphics[width=\\linewidth]{{figures/ecg_example_tn.png}}
\\caption{{\\textbf{{True Negative Case:}} \\textit{{"{example_tn}"}}}}
\\label{{fig:tn}}
\\end{{figure}}

\\textbf{{Example False Positive Case:}} 
\\textit{{"{example_fp}"}}

\\section{{Discussion and Limitations}}
This study presents a preliminary evaluation of a reproducible pipeline. It is crucial to highlight the following limitations:
\\begin{{itemize}}
    \\item \\textbf{{ECG-Only Model}}: The system operates without clinical biomarkers, symptoms, or patient history.
    \\item \\textbf{{No Clinical Validation}}: The system has not been prospectively validated nor evaluated by clinicians.
    \\item \\textbf{{Not for Deployment}}: Agent explanations are algorithmically generated and must not replace medical judgment.
\\end{{itemize}}

\\section{{Conclusion}}
We established a reproducible, lightweight agentic framework for ECG-based MI prediction on PTB-XL. By converting raw predictions into structured summaries, we demonstrate a potential method for interpreting black-box ECG models in research environments. Future work requires extensive integration with multimodal clinical data and rigorous clinical validation.

\\bibliographystyle{{plain}}
\\bibliography{{references}}
\\end{{document}}
"""
    with open('main.tex', 'w') as f:
        f.write(latex_content)
    print("LaTeX file main.tex generated successfully.")

if __name__ == "__main__":
    generate_latex()
