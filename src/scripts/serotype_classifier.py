import json
import argparse
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

from models import TransformerLRClassifier
from consts import DEFAULT_SEP, DEFAULT_BATCH_SIZE, DEFAULT_MISSING_LABEL


def main(args):
    device = torch.device(args.device)

    sep = args.model_params.get("sep", DEFAULT_SEP)
    missing_label = args.model_params.get("missing_label", DEFAULT_MISSING_LABEL)
    
    print(f"Loading embeddings from: {args.embeddings}")
    X = np.load(args.embeddings, allow_pickle=True)  # shape: (N, L, D)

    # Parse sample IDs and extract labels
    sample_keys = list(X.keys())
    sample_keys = [key for key in sample_keys if key.startswith('cbl')]  # Filter to only capsulated samples TODO
    labels_data = []
    for key in sample_keys:
        parts = key.split(sep)
        if len(parts) >= 2:
            capsule_label = parts[0]
            sample_id = sep.join(parts[1:])  # In case sample ID contains the separator
            labels_data.append({
                'key': key,
                'capsule_label': capsule_label,
                'sample_id': sample_id
            })
        else:
            print(f"Warning: Skipping malformed key: {key}")
    
    labels_df = pd.DataFrame(labels_data)
    labels_df['is_capsule'] = labels_df['capsule_label'].map(lambda x: 1 if x == "cbl" else 0).astype(int)
    
    # Load true serotype labels if provided
    if args.true_labels:
        print(f"Loading true labels from: {args.true_labels}")
        true_labels = pd.read_csv(args.true_labels, sep="\t", index_col=0)
        
        # Match sample IDs with true labels
        labels_df = labels_df.merge(
            true_labels, 
            left_on='sample_id', 
            right_index=True, 
            how='inner'
        )
        
        # Filter to only capsulated samples for serotype classification
        capsule_mask = labels_df['is_capsule'] == 1
        if not capsule_mask.any():
            print("No capsulated samples found. Cannot perform serotype classification.")
            return
            
        labels_df = labels_df[capsule_mask].copy()
        print(f"Found {len(labels_df)} capsulated samples for serotype classification")
    else:
        print("No true labels provided. Will only extract predictions without evaluation.")
        # Filter to only capsulated samples
        capsule_mask = labels_df['is_capsule'] == 1
        labels_df = labels_df[capsule_mask].copy()
        print(f"Found {len(labels_df)} capsulated samples")
    
    # Get embeddings for selected samples
    X_filtered = np.stack([X[key] for key in labels_df['key']])
    
    print(f"Loading model from: {args.model}")
    model_save_dict = torch.load(args.model, map_location=device)
    model_config = model_save_dict['model_config']
    serotype_to_idx = model_save_dict['serotype_to_idx']
    num_serotypes = model_save_dict['num_serotypes']
    
    print(f"Model configuration: {model_config}")
    print(f"Number of serotypes: {num_serotypes}")
    
    # Initialize model with saved configuration
    model = TransformerLRClassifier(
        input_dim=model_config['input_dim'],
        num_classes=model_config['num_classes'],
        output_dim=model_config['output_dim'],
        nhead=model_config['nhead'],
        num_layers=model_config['num_layers']
    ).to(device)
    
    # Load the model state
    model.load_state_dict(model_save_dict['model_state_dict'])
    model.eval()

    # Create reverse mapping for predictions
    idx_to_serotype = {v: k for k, v in serotype_to_idx.items()}

    print("Running serotype inference...")
    all_predictions = []
    all_probabilities = []
    all_cbl_predictions = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(X_filtered), args.batch_size)):
            # The embeddings are already the final projected embeddings (z) from the model
            batch = torch.tensor(X_filtered[i:i+args.batch_size], dtype=torch.float32, device=device)
            
            # Apply the classifier layers directly to the embeddings
            cbl_logits = model.cbl_classifier(batch)
            serotype_logits = model.serotype_classifier(batch)
            
            # CBL predictions
            cbl_probs = torch.softmax(cbl_logits, dim=1)[:, 1].cpu().numpy()  # Probability of being capsulated
            all_cbl_predictions.extend(cbl_probs)
            
            # Serotype predictions
            serotype_probs = torch.softmax(serotype_logits, dim=1).cpu().numpy()
            predicted_indices = torch.argmax(serotype_logits, dim=1).cpu().numpy()
            
            # Convert indices to serotype names
            batch_predictions = [idx_to_serotype[idx] for idx in predicted_indices]
            all_predictions.extend(batch_predictions)
            all_probabilities.extend(serotype_probs)
    
    # Add predictions to dataframe
    labels_df['predicted_serotype'] = all_predictions
    labels_df['cbl_probability'] = all_cbl_predictions
    
    # Add individual serotype probabilities
    all_probabilities = np.array(all_probabilities)
    for i, serotype in enumerate(sorted(serotype_to_idx.keys())):
        labels_df[f'prob_{serotype}'] = all_probabilities[:, serotype_to_idx[serotype]]
    
    predictions_output = f"{args.output_dir}/serotype_predictions.txt"
    labels_df.to_csv(predictions_output, sep='\t', index=False)
    print(f"Predictions saved to: {predictions_output}")
    
    report_path = f"{args.output_dir}/classification_report.txt"
    if args.true_labels and 'Serotype' in labels_df.columns:
        y_true = labels_df['Serotype'].values
        y_pred = labels_df['predicted_serotype'].values
        
        valid_mask = y_true != missing_label
        if not valid_mask.all():
            print(f"Removing {(~valid_mask).sum()} samples with missing true labels")
            y_true = y_true[valid_mask]
            y_pred = y_pred[valid_mask]
            labels_df_eval = labels_df[valid_mask].copy()
        else:
            labels_df_eval = labels_df.copy()
        
        if len(y_true) == 0:
            print("No samples with valid true labels found for evaluation.")
            return
        
        print(f"Evaluating on {len(y_true)} samples with valid labels")
        accuracy = accuracy_score(y_true, y_pred)
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        
        # Generate classification reports
        unique_serotypes = sorted(list(set(y_true) | set(y_pred)))
        clf_report = classification_report(y_true, y_pred, target_names=unique_serotypes, labels=unique_serotypes)
        
        cm = confusion_matrix(y_true, y_pred, labels=unique_serotypes)
        conf_matrix = pd.DataFrame(cm, index=unique_serotypes, columns=unique_serotypes)
        conf_matrix.to_csv(f"{args.output_dir}/confusion_matrix_df.csv")
        
        # Calculate per-serotype confidence statistics
        confidence_stats = []
        for serotype in unique_serotypes:
            if serotype in serotype_to_idx:
                serotype_mask = labels_df_eval['predicted_serotype'] == serotype
                if serotype_mask.any():
                    confidences = labels_df_eval.loc[serotype_mask, f'prob_{serotype}'].values
                    confidence_stats.append({
                        'serotype': serotype,
                        'count': serotype_mask.sum(),
                        'mean_confidence': np.mean(confidences),
                        'std_confidence': np.std(confidences),
                        'min_confidence': np.min(confidences),
                        'max_confidence': np.max(confidences)
                    })
        
        confidence_df = pd.DataFrame(confidence_stats)
        if not confidence_df.empty:
            confidence_output = f"{args.output_dir}/serotype_confidence_stats.tsv"
            confidence_df.to_csv(confidence_output, sep='\t', index=False)
            print(f"Confidence statistics saved to: {confidence_output}")
        
        print("Serotype Classification Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score (weighted): {f1_weighted:.4f}")
        print(f"F1 Score (macro): {f1_macro:.4f}")
        print("\nClassification Report:")
        print(clf_report)
        
        with open(report_path, 'w') as f:
            f.write("Serotype Classification Results\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Total samples evaluated: {len(y_true)}\n")
            f.write(f"Number of unique serotypes: {len(unique_serotypes)}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"F1 Score (weighted): {f1_weighted:.4f}\n")
            f.write(f"F1 Score (macro): {f1_macro:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(clf_report)
            
            if not confidence_df.empty:
                f.write("\n\nPer-Serotype Confidence Statistics:\n")
                f.write(confidence_df.to_string(index=False))
        
        print(f"Results saved to: {args.output_dir}")
        
    else:
        print("No evaluation performed (no true labels provided or found).")
        with open(report_path, 'w') as f:
            f.write("Serotype Prediction Results\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Total samples processed: {len(labels_df)}\n")
            f.write(f"Number of predicted serotypes: {len(set(all_predictions))}\n")
            f.write(f"Predicted serotypes: {', '.join(sorted(set(all_predictions)))}\n\n")
            f.write("Prediction counts:\n")
            pred_counts = pd.Series(all_predictions).value_counts()
            f.write(pred_counts.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate serotype classification using TransformerLRClassifier")
    parser.add_argument("--embeddings", required=True, 
                        help="Path to .npz embeddings file containing sample embeddings")
    parser.add_argument("--model", required=True,
                        help="Path to the saved TransformerLRClassifier model file (.pth)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for classification report")
    parser.add_argument("--true_labels", 
                        help="TSV file with true serotype labels indexed by sample ID")
    parser.add_argument("--device", default="cpu",
                        help="Device to use for inference (cpu or cuda)")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Batch size for inference")
    parser.add_argument("--model_params", type=str, default="{}",
                        help="JSON string of model parameters")

    args = parser.parse_args()
    
    try:
        args.model_params = json.loads(args.model_params)
        if not isinstance(args.model_params, dict):
            print("Model parameters should be a JSON object.")
            args.model_params = {}
    except json.JSONDecodeError:
        print("Error parsing model parameters JSON string.")
        args.model_params = {}
    finally:
        print("Model parameters:", args.model_params)

    main(args)
