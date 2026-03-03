import os
import argparse
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import pytorch_lightning as pl
from regression_model.model import dmpnn
from features.date_pre import MoleculeDataset, collate_fn
from pytorch_lightning import Trainer, LightningModule

class InferenceModule(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batched_graph, extra_features):
        return self.model(batched_graph, extra_features)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        batched_graph, extra_features = batch
        outputs = self(batched_graph, extra_features).squeeze(-1)
        return outputs

def load_model(checkpoint_path, model_cls):

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    renamed_count = 0
    
    for key, value in state_dict.items():
        if '_solute' in key:
            new_key = key.replace('_solute', '')
            new_state_dict[new_key] = value
            renamed_count += 1
        else:
            new_state_dict[key] = value
    
    if renamed_count > 0:
        print(f"✓ Renamed {renamed_count} parameter keys (_solute → '')")
    
    
    trained_model = InferenceModule(model_cls)
    trained_model.load_state_dict(new_state_dict)
    trained_model.eval()
    trained_model.freeze()
    
    return trained_model

def parse_args():
    
    # Model  parameters
    parser.add_argument('--node_feat_dim', type=int, default=109)
    parser.add_argument('--edge_feat_dim', type=int, default=13)
    parser.add_argument('--edge_output_dim', type=int, default=400)
    parser.add_argument('--node_output_dim', type=int, default=400)
    parser.add_argument('--extra_dim', type=int, default=26)
    parser.add_argument('--num_rounds', type=int, default=7)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--moe_hid_dim', type=int, default=400)
    parser.add_argument('--num_heads', type=int, default=8)
    
    # File paths
    parser.add_argument('--checkpoint_path', type=str, 
                        required=True,
                        help='Path to model checkpoint file')
    parser.add_argument('--input_csv', type=str,
                        required=True,
                        help='Path to input CSV file with SMILES and features')
    parser.add_argument('--output_csv', type=str,
                        required=True,
                        help='Path to save predictions')
    
 
    parser.add_argument('--smiles_column', type=str, default='SMILES')
    parser.add_argument('--feature_start_col', type=int, default=1)
    parser.add_argument('--standardize', action='store_true', default=True)
    parser.add_argument('--no_standardize', dest='standardize', 
                        action='store_false')
    parser.add_argument('--prediction_column', type=str, default='Predicted',
                        help='Name of prediction column in output CSV')
    
  
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--num_workers', type=int, default=11)
    
 
    parser.add_argument('--accelerator', type=str, default='auto',
                        choices=['auto', 'gpu', 'cpu'])
    parser.add_argument('--devices', type=int, default=1)
    
    return parser.parse_args()

def main():
    args = parse_args()
  
    print("=" * 60)
    print("=" * 60)
    for arg, value in sorted(vars(args).items()):
        print(f"{arg:25s}: {value}")
    print("=" * 60)
    
    
    print("\nInitializing model...")
    model = dmpnn(
        node_feat_dim=args.node_feat_dim,
        edge_feat_dim=args.edge_feat_dim,
        edge_output_dim=args.edge_output_dim,
        node_output_dim=args.node_output_dim,
        extra_dim=args.extra_dim,
        num_rounds=args.num_rounds,
        dropout_rate=args.dropout_rate,
        num_experts=args.num_experts,
        moe_hid_dim=args.moe_hid_dim,
        num_heads=args.num_heads
    )
    
    # Check if checkpoint exists
    if not Path(args.checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    
  
    print(f"\nLoading model from: {args.checkpoint_path}")
    trained_model = load_model(args.checkpoint_path, model)
    print("Model loaded successfully!")
    
    
    if not Path(args.input_csv).exists():
        raise FileNotFoundError(f"Input file not found: {args.input_csv}")
    
   
    print(f"\nLoading data from: {args.input_csv}")
    pre_df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(pre_df)} samples")
    
    # Validate SMILES column
    if args.smiles_column not in pre_df.columns:
        raise ValueError(
            f"Column '{args.smiles_column}' not found in CSV. "
            f"Available columns: {list(pre_df.columns)}"
        )
    
    
    smiles_list = pre_df[args.smiles_column].tolist()
    
  
    feature_columns = pre_df.columns[args.feature_start_col:]
    print(f"Number of feature columns: {len(feature_columns)}")
    

    if len(feature_columns) != args.extra_dim:
        print(f"WARNING: Feature dimension mismatch!")
        print(f"  CSV has {len(feature_columns)} features")
        print(f"  Model expects {args.extra_dim} features")
        raise ValueError("Feature dimension mismatch")
    

    if args.standardize:
        print("\nApplying standardization to features...")
        scaler = StandardScaler()
        pre_df[feature_columns] = scaler.fit_transform(pre_df[feature_columns])
        print("Standardization completed")
    else:
        print("\nSkipping standardization (using raw features)")
    
    # Convert features to tensor
    extra_features = torch.tensor(
        pre_df[feature_columns].values, 
        dtype=torch.float32
    )
    
    # Create dataset
    print("\nCreating dataset...")
    predict_dataset = MoleculeDataset(
        smiles_list,
        extra_features
    )
    print(f"Dataset size: {len(predict_dataset)}")
    
 
    predict_dataloader = DataLoader(
        predict_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    # train
    if args.accelerator == 'auto':
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    else:
        accelerator = args.accelerator
    
    devices = args.devices if accelerator == 'gpu' else None
    
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices
    )
    

    print(f"\nRunning inference on {accelerator.upper()}...")
    print(f"Batch size: {args.batch_size}")
    predictions = trainer.predict(trained_model, dataloaders=predict_dataloader)
    
    # Collect predictions
    print("\nCollecting predictions...")
    output_list = []
    for batch in predictions:
        output_list.extend(batch.tolist())
    
    print(f"Total predictions: {len(output_list)}")
    
    # Add predictions to dataframe
    pre_df[args.prediction_column] = output_list
    
    # Create output directory
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save results
    pre_df.to_csv(args.output_csv, index=False)
    print(f"\n{'=' * 60}")
    print(f"Predictions saved to: {args.output_csv}")
    print(f"{'=' * 60}")
    
if __name__ == "__main__":
    main()
