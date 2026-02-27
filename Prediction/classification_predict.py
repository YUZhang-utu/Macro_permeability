import pandas as pd
import numpy as np
import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, LightningModule
from features.date_pre import MoleculeDataset, collate_fn
from Multi_DDPP.model import dmpnn

class InferenceModule(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batched_graph, extra_features):
        return self.model(batched_graph, extra_features)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        batched_graph, extra_features = batch
        outputs = self(batched_graph, extra_features)
        return torch.sigmoid(outputs).squeeze(-1)

def load_model(checkpoint_path, model_cls):
    trained_model = InferenceModule.load_from_checkpoint(
        checkpoint_path, 
        model=model_cls, 
        strict=False
    )
    trained_model.eval()
    trained_model.freeze()
    return trained_model

def parse_args():
    parser = argparse.ArgumentParser(description='Classification Prediction')
    # Model  parameters
    parser.add_argument('--node_feat_dim', type=int, default=109)
    parser.add_argument('--edge_feat_dim', type=int, default=13)
    parser.add_argument('--edge_output_dim', type=int, default=400)
    parser.add_argument('--node_output_dim', type=int, default=400)
    parser.add_argument('--extra_dim', type=int, default=19)
    parser.add_argument('--num_rounds', type=int, default=7)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--moe_hid_dim', type=int, default=400)
    
    # File paths
    parser.add_argument('--checkpoint_path', type=str, 
                        required=True)
    parser.add_argument('--input_csv', type=str,
                        required=True)
    parser.add_argument('--output_csv', type=str,
                        required=True)
    
    
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--num_workers', type=int, default=11)
    parser.add_argument('--smiles_column', type=str, default='SMILES')
    parser.add_argument('--feature_start_col', type=int, default=1)
    
    
    parser.add_argument('--accelerator', type=str, default='auto',
                        choices=['auto', 'gpu', 'cpu'])
    parser.add_argument('--devices', type=int, default=1)
    
    return parser.parse_args()

def main():
    
    args = parse_args()
    print("=" * 50)
    print("Inference Configuration:")
    print("=" * 50)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("=" * 50)
    
    
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
    )
    
    
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
    
    # Extract SMILES and features
    if args.smiles_column not in pre_df.columns:
        raise ValueError(f"Column '{args.smiles_column}' not found in CSV")
    
    smiles_list = pre_df[args.smiles_column].tolist()
    feature_columns = pre_df.columns[args.feature_start_col:]
    extra_features = torch.tensor(
        pre_df[feature_columns].values, 
        dtype=torch.float32
    )
    
    print(f"Number of features: {len(feature_columns)}")
    
    
    predict_dataset = MoleculeDataset(
        smiles_list,
        extra_features
    )
    
    # Create data loader
    predict_dataloader = DataLoader(
        predict_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False
    )
    

    if args.accelerator == 'auto':
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    else:
        accelerator = args.accelerator
    
    devices = args.devices if accelerator == 'gpu' else None
    
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices
    )
    

    print(f"\nRunning inference on {accelerator.upper()}...")
    predictions = trainer.predict(trained_model, dataloaders=predict_dataloader)
    
    # Collect predictions
    output_list = []
    for batch in predictions:
        output_list.extend(batch.tolist())
    
  
    pre_df['predictions'] = output_list
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save results
    pre_df.to_csv(args.output_csv, index=False)
    print(f"\nPredictions saved to: {args.output_csv}")
    print(f"Total predictions: {len(output_list)}")
    print("\nInference completed successfully!")

if __name__ == "__main__":
    main()
