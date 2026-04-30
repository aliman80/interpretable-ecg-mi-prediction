import os
import argparse
from src.data_loader import prepare_data, download_ptbxl_csvs
from src.train import train_model
from src.evaluate import evaluate_model
from src.agents import run_error_analysis

def main():
    parser = argparse.ArgumentParser(description="Run ECG MI Classification Pipeline")
    parser.add_argument('--max_samples', type=int, default=None, help='Limit number of samples for quick testing')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    args = parser.parse_args()

    print("1. Setting up data...")
    download_ptbxl_csvs()
    X_train, y_train, X_val, y_val, X_test, y_test, df_test = prepare_data(max_samples=args.max_samples)
    
    print(f"Data shapes: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
    
    print("\n2. Training model...")
    model = train_model(X_train, y_train, X_val, y_val, 
                        epochs=args.epochs, batch_size=32, lr=1e-3, 
                        device=args.device, save_path='results/best_model.pth')
                        
    print("\n3. Evaluating model...")
    evaluate_model(X_test, y_test, df_test, model_path='results/best_model.pth', device=args.device)
    
    print("\n4. Running Agentic Error Analysis...")
    run_error_analysis('results/test_predictions.csv')
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()
