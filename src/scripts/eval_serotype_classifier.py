import json
import argparse
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

from .models import ModelRegistry
from .consts import (
    DEFAULT_LABEL_COLUMN, DEFAULT_NONCBL_LABEL, DEFAULT_SEP,
    DEFAULT_BATCH_SIZE, DEFAULT_MISSING_LABEL, CONTIG_SEP,
    DEFAULT_ENERGY_TEMPERATURE, DEFAULT_HEAD_MODEL
)
from .utils import get_sample_id


def main(args):
    device = torch.device(args.device)

    sep = args.model_params.get("sep", DEFAULT_SEP)
    label_column = args.model_params.get("label_column", DEFAULT_LABEL_COLUMN)
    missing_label = args.model_params.get("missing_label", DEFAULT_MISSING_LABEL)
    noncbl_label = args.model_params.get("noncbl_label", DEFAULT_NONCBL_LABEL)
    head_model = args.model_params.get("head_model", DEFAULT_HEAD_MODEL)
    
    print(f"Loading embeddings and labels")
    X = np.load(args.embeddings, allow_pickle=True)  # shape: (N, L, D)
    labels_df = pd.read_csv(args.labels, index_col=0, sep="\t" if args.labels.endswith(".tsv") else ",")
    labels_df['Serotype'] = labels_df[label_column].fillna(missing_label)  # TODO should be empty already
    labels_df = labels_df[labels_df["Serotype"] != missing_label]

    keys = labels_df["Is_capsule"].map(lambda x: "cbl" if x else "non-cbl") + sep + get_sample_id(labels_df)
    X_filtered = np.stack([X[key] for key in keys])
    print(f"Loaded {len(X_filtered)} embeddings for capsulated samples")

    print(f"Loading model from: {args.model}")
    model_save_dict = torch.load(args.model, map_location=device)
    model_config = model_save_dict['model_config']
    serotype_to_idx = model_save_dict['serotype_to_idx']
    num_serotypes = model_save_dict['num_serotypes']
    
    print(f"Model configuration: {model_config}")
    print(f"Number of serotypes: {num_serotypes}")
    
    # Initialize model with saved configuration
    model = ModelRegistry.get_model_class(head_model) \
        .from_config(model_config) \
        .to(device)
    model.load_state_dict(model_save_dict['model_state_dict'])
    model.eval()

    # Create reverse mapping for predictions
    idx_to_serotype = {v: k for k, v in serotype_to_idx.items()}

    print("Running serotype inference...")
    all_predictions = []
    all_probabilities = []
    all_cbl_predictions = []

    energy_temperature = args.energy_temperature
    all_serotype_energies = []
    
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

            # Serotype energy (for OOD): E = -T * logsumexp(logits/T)
            if args.collect_energies:
                energies = (-energy_temperature * torch.logsumexp(serotype_logits / energy_temperature, dim=-1)).cpu().numpy()
                all_serotype_energies.append(energies)
    
    # Add predictions to dataframe
    labels_df['predicted_serotype'] = all_predictions
    labels_df['cbl_probability'] = all_cbl_predictions
    
    # Add individual serotype probabilities
    all_probabilities = np.array(all_probabilities)
    for i, serotype in enumerate(sorted(serotype_to_idx.keys())):
        labels_df[f'prob_{serotype}'] = all_probabilities[:, serotype_to_idx[serotype]]
    
    # Dump energies if requested
    if args.collect_energies and len(all_serotype_energies) > 0:
        serotype_energies = np.concatenate(all_serotype_energies)
        # Attach identifiers and minimal metadata
        labels_df['sample_id'] = get_sample_id(labels_df)
        labels_df['energy_serotype'] = serotype_energies
        # Persist CSV
        energy_cols = ['sample_id', 'Serotype', 'predicted_serotype', 'Is_capsule', 'energy_serotype']
        energies_path = f"{args.output_dir}/serotype_energies.csv"
        labels_df.to_csv(energies_path, columns=energy_cols, index=False)
        print(f"Serotype energies saved to: {energies_path}")
        # Optional summary JSON
        if args.save_energy_summary:
            try:
                caps_mask = labels_df['Is_capsule'].astype(bool)
                caps_energies = labels_df.loc[caps_mask, 'energy_serotype'].values
                percentiles = {p: float(np.percentile(caps_energies, p)) for p in [93.0, 95.0, 99.0, 99.5] if len(caps_energies) > 0}
                summary = {
                    'temperature': energy_temperature,
                    'count_total': int(len(labels_df)),
                    'count_capsulated': int(caps_mask.sum()),
                    'percentiles_serotype': percentiles,
                }
                energy_summary_path = f"{args.output_dir}/energy_summary.json"
                with open(energy_summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                print(f"Energy summary saved to: {energy_summary_path}")
            except Exception as e:
                print(f"Failed to save energy summary JSON: {e}")
    
    predictions_output = f"{args.output_dir}/serotype_predictions.txt"
    labels_df.to_csv(predictions_output, sep='\t', index=False)
    print(f"Predictions saved to: {predictions_output}")
    
    report_path = f"{args.output_dir}/classification_report.txt"
    # Exclude NON-CBL
    labels_df = labels_df[labels_df["Serotype"] != noncbl_label].copy()
    if args.labels and 'Serotype' in labels_df.columns:
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
        
        # Skip samples whose true label is not a resolved serotype class
        # (e.g. serogroup-only labels like "Serogroup 24" or compound labels)
        known_mask = np.array([lbl in serotype_to_idx for lbl in y_true])
        if not known_mask.all():
            n_dropped = (~known_mask).sum()
            dropped_labels = sorted(set(y_true[~known_mask]))
            print(f"Removing {n_dropped} samples with true labels not in serotype_to_idx: {dropped_labels}")
            y_true = y_true[known_mask]
            y_pred = y_pred[known_mask]
            labels_df_eval = labels_df_eval[known_mask].copy()
        
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
            f.write(str(clf_report))
            
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
    parser = argparse.ArgumentParser(description="Evaluate serotype classification")
    parser.add_argument("--embeddings", required=True, 
                        help="Path to .npz embeddings file containing sample embeddings")
    parser.add_argument("--labels", 
                        help="TSV file with true serotype labels indexed by sample ID")
    parser.add_argument("--model", required=True,
                        help="Path to the saved model file (.pth)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for classification report")
    parser.add_argument("--device", default="cpu",
                        help="Device to use for inference (cpu or cuda)")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Batch size for inference")
    parser.add_argument("--model_params", type=str, default="{}",
                        help="JSON string of model parameters")
    parser.add_argument("--collect_energies", action="store_true",
                        help="Optional flag to dump per-sample serotype energies for ID calibration")
    parser.add_argument("--energy_temperature", type=float, default=DEFAULT_ENERGY_TEMPERATURE,
                        help="Temperature T used in energy calculation")
    parser.add_argument("--save_energy_summary", action="store_true",
                        help="Optional flag to save percentile summary of serotype energies")

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
