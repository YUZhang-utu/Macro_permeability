import os
import argparse
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchmetrics import AUROC, MatthewsCorrCoef, Accuracy
from Multi_DDPP.model import dmpnn
from features.data import MoleculeDataset, collate_fn


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LitGraphModel(pl.LightningModule):
    def __init__(self, model, teacher_model, train_path, val_path, learning_rate, lambda_=0.2, temperature=6.0):
        super().__init__()
        self.model = model
        self.teacher_model = teacher_model  # add teacher model
        self.learning_rate = learning_rate
        self.train_path = train_path
        self.val_path = val_path
        self.lambda_ = lambda_  # weitht of soft label 
        self.temperature = temperature  # T
        
        # freeze teacher model
        if self.teacher_model is not None:
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            self.teacher_model.eval()

        
        self.train_accuracy = Accuracy(task='binary')
        self.train_auc = AUROC(task='binary')
        self.train_mcc = MatthewsCorrCoef(task='binary', num_classes=2)

    def forward(self, batched_graph, extra_features):
        return self.model(batched_graph, extra_features)

    def training_step(self, batch, batch_idx):
        batched_graph, targets, extra_features = batch

        
        student_logits = self(batched_graph, extra_features).view(-1)

        # hard label loss
        hard_loss = F.binary_cross_entropy_with_logits(student_logits, targets.float())

        # KD
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_logits = self.teacher_model(batched_graph, extra_features).view(-1)
            
            # smooth distribution of soft label
            # p(z, T) = σ(z/T)
            student_soft = torch.sigmoid(student_logits / self.temperature)
            teacher_soft = torch.sigmoid(teacher_logits / self.temperature)
            
            
            # L_D = T^2 * MSE(p_s^soft, p_t^soft)
            soft_loss = F.mse_loss(student_soft, teacher_soft) * (self.temperature ** 2)
            
            # total：L_total = L_hard + λ * L_D
            total_loss = hard_loss + self.lambda_ * soft_loss
            
            
            self.log('train_hard_loss', hard_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('train_soft_loss', soft_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        batched_graph, targets, extra_features = batch
        predictions = self(batched_graph, extra_features)
        predictions = predictions.view(-1)
        loss = F.binary_cross_entropy_with_logits(predictions, targets.float())

        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {'val_loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_df = pd.read_csv(self.train_path)
            val_df = pd.read_csv(self.val_path)
            self.train_dataset = MoleculeDataset(train_df['SMILES'].tolist(),
                                                 train_df['Label'].values,
                                                 torch.tensor(train_df.iloc[:, 2:].values, dtype=torch.float32))
            self.val_dataset = MoleculeDataset(val_df['SMILES'].tolist(),
                                               val_df['Label'].values,
                                               torch.tensor(val_df.iloc[:, 2:].values, dtype=torch.float32))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=24, shuffle=True, collate_fn=collate_fn, num_workers=11)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=24, shuffle=False, collate_fn=collate_fn, num_workers=11)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Knowledge Distillation Training for Molecular Property Prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data parameters
    parser.add_argument('--small_data_path', type=str, required=True)
    parser.add_argument('--teacher_model_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='output')
    
    # Model  parameters
    parser.add_argument('--node_feat_dim', type=int, default=109)
    parser.add_argument('--edge_feat_dim', type=int, default=13)
    parser.add_argument('--edge_output_dim', type=int, default=400)
    parser.add_argument('--node_output_dim', type=int, default=400)
    parser.add_argument('--extra_dim', type=int, default=19)
    parser.add_argument('--num_rounds', type=int, default=7)
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--moe_hid_dim', type=int, default=400)
    
    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--num_workers', type=int, default=11)
    parser.add_argument('--patience', type=int, default=30)
    
    # Knowledge distillation parameters
    parser.add_argument('--lambda_kd', type=float, default=0.2)
    parser.add_argument('--temperature', type=float, default=6.0)
    parser.add_argument('--teacher_dropout', type=float, default=0.2)
    parser.add_argument('--student_dropout', type=float, default=0.2)
    
    # Train/test split parameters
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--split_random_state', type=int, default=2)
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--accelerator', type=str, default='gpu', choices=['gpu', 'cpu'])
    parser.add_argument('--devices', type=int, default=1)
    
    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)

    # Load student dataset
    student_df = pd.read_csv(args.small_data_path)

    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'Output directory: {args.output_dir}')

    # Determine teacher model path
    if args.teacher_model_path is None:
        teacher_model_path = os.path.join(os.path.dirname(args.small_data_path), 'teacher_model.ckpt')
    else:
        teacher_model_path = args.teacher_model_path
    
    print(f'\nTeacher model path: {teacher_model_path}')

    # Load teacher model if exists
    if os.path.exists(teacher_model_path):
        teacher_model = dmpnn(
            node_feat_dim=args.node_feat_dim,
            edge_feat_dim=args.edge_feat_dim,
            edge_output_dim=args.edge_output_dim,
            node_output_dim=args.node_output_dim,
            extra_dim=args.extra_dim,
            num_rounds=args.num_rounds,
            dropout_rate=args.teacher_dropout,
            num_experts=args.num_experts,
            moe_hid_dim=args.moe_hid_dim
        )

        try:
            teacher_model.load_state_dict(torch.load(teacher_model_path), strict=False)
            print('Teacher model loaded successfully!')
        except Exception as e:
            print(f'Error loading teacher model: {e}')
            print('Training without knowledge distillation...')
            teacher_model = None
    else:
        print(f'Teacher model not found at {teacher_model_path}')
        print('Training without knowledge distillation...')
        teacher_model = None

    # Train/test split
    print(f'\n{"="*70}')
    print(f'Splitting dataset into train and test sets')
    print(f'{"="*70}')
    
    unique_smiles = student_df['SMILES'].unique()
    train_smiles, test_smiles = train_test_split(
        unique_smiles, 
        test_size=args.test_size,
        random_state=args.split_random_state,
        shuffle=True
    )
    
    train_indices = student_df[student_df['SMILES'].isin(train_smiles)].index
    test_indices = student_df[student_df['SMILES'].isin(test_smiles)].index

    train_df = student_df.iloc[train_indices]
    test_df = student_df.iloc[test_indices]
    
    print(f'Train size: {len(train_df)}, Test size: {len(test_df)}')

    # Save data splits
    train_df.to_csv(os.path.join(args.output_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(args.output_dir, 'test.csv'), index=False)

    # Initialize student model
    student_model = dmpnn(
        node_feat_dim=args.node_feat_dim,
        edge_feat_dim=args.edge_feat_dim,
        edge_output_dim=args.edge_output_dim,
        node_output_dim=args.node_output_dim,
        extra_dim=args.extra_dim,
        num_rounds=args.num_rounds,
        dropout_rate=args.student_dropout,
        num_experts=args.num_experts,
        moe_hid_dim=args.moe_hid_dim
    )

    student_module = LitGraphModel(
        student_model, 
        teacher_model,
        os.path.join(args.output_dir, 'train.csv'),
        os.path.join(args.output_dir, 'test.csv'),
        learning_rate=args.learning_rate,
        lambda_=args.lambda_kd,
        temperature=args.temperature
    )

    # Setup training
    logger = TensorBoardLogger(os.path.join(args.output_dir, 'student_logs'), name='logs_student')
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, 'student_model'),
        filename='best-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        verbose=True
    )
    
    early_stopping_callback = EarlyStopping(
        monitor='val_loss', 
        patience=args.patience, 
        verbose=True,
        mode='min'
    )

    trainer = Trainer(
        max_epochs=args.max_epochs, 
        logger=logger, 
        callbacks=[checkpoint_callback, early_stopping_callback],
        accelerator=args.accelerator, 
        devices=args.devices,
        deterministic=True,
        enable_progress_bar=True
    )

    print(f'\n{"="*70}')
    print(f'Starting training')
    print(f'{"="*70}')
    
    trainer.fit(student_module)
    
    print(f'\n{"="*70}')
    print(f'Training completed successfully!')
    print(f'{"="*70}')


if __name__ == "__main__":
    main()
