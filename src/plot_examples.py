import wfdb
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_12_lead_ecg(record_name, pn_dir, title, save_path):
    print(f"Fetching {record_name} from {pn_dir}...")
    try:
        record = wfdb.rdsamp(record_name, pn_dir=pn_dir)
        signals = record[0] # (length, 12)
        
        # Define leads for 4x3 grid
        leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
        fig, axes = plt.subplots(4, 3, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        for i, ax in enumerate(axes.flatten()):
            if i < 12:
                # Plot 10 seconds of signal (1000 samples at 100Hz)
                ax.plot(signals[:1000, i], color='black', linewidth=1)
                ax.set_title(f"Lead {leads[i]}")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(True, linestyle='--', alpha=0.5)
                
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {save_path}")
    except Exception as e:
        print(f"Failed to plot {record_name}: {e}")

if __name__ == "__main__":
    # Generic Normal ECG (TN)
    # Using record 00001_lr (NORM)
    plot_12_lead_ecg(
        record_name='00001_lr', 
        pn_dir='ptb-xl/1.0.3/records100/00000',
        title='12-Lead ECG: Normal Sinus Rhythm (True Negative Example)',
        save_path='../paper/figures/ecg_example_tn.png'
    )
    
    # Generic MI ECG (TP)
    # Using record 00004_lr (MI)
    plot_12_lead_ecg(
        record_name='00004_lr', 
        pn_dir='ptb-xl/1.0.3/records100/00000',
        title='12-Lead ECG: Myocardial Infarction (True Positive Example)',
        save_path='../paper/figures/ecg_example_tp.png'
    )
