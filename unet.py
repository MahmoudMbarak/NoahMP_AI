import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import logging
import sys
from datetime import datetime
import xarray as xr
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import r2_score
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import json

# ── User configuration ─────────────────────────────────────────────────────
# Directory with one subdirectory per ensemble member (sim001, sim002, ...),
# each holding a processed_surface_data.nc file. Override with the DATA_DIR env var.
DATA_DIR = os.environ.get('DATA_DIR', '/path/to/processed_ensemble')
# Train on ensemble members 1..MAX_ENSEMBLE_MEMBER only; set to None to use all.
MAX_ENSEMBLE_MEMBER = 560

class WarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, warmup_lr=0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.warmup_lr = warmup_lr
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * epoch / self.warmup_epochs
        else:
            # Cosine decay after warmup
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1 + np.cos(np.pi * progress))
        
        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class SoilMoistureUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder path
        self.enc1 = DoubleConv(3, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder path
        self.dec4 = DoubleConv(1024 + 512, 512)
        self.dec3 = DoubleConv(512 + 256, 256)
        self.dec2 = DoubleConv(256 + 128, 128)
        self.dec1 = DoubleConv(128 + 64, 64)

        # Final convolution with ReLU for non-negative output
        self.final_conv = nn.Sequential(
            nn.Conv3d(64, 1, kernel_size=1),
            nn.ReLU()
        )
        
        # Spatial pooling
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def forward(self, x):
        # Encoder path with skip connections
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder path
        dec4 = self.dec4(torch.cat([F.interpolate(bottleneck, size=enc4.shape[2:], mode='nearest'), enc4], dim=1))
        dec3 = self.dec3(torch.cat([F.interpolate(dec4, size=enc3.shape[2:], mode='nearest'), enc3], dim=1))
        dec2 = self.dec2(torch.cat([F.interpolate(dec3, size=enc2.shape[2:], mode='nearest'), enc2], dim=1))
        dec1 = self.dec1(torch.cat([F.interpolate(dec2, size=enc1.shape[2:], mode='nearest'), enc1], dim=1))

        # Final convolution 
        output = self.final_conv(dec1)
        
        return output

class SoilMoistureEnsembleDataset(Dataset):
    def __init__(self, base_dir, split='train', seed=42, indices_file='ensemble_split_indices.json'):
        logging.info(f"{'==='*20}")
        logging.info(f"Initializing {split.capitalize()} Dataset")
        
        self.base_dir = Path(base_dir)
        self.file_paths = []
        
        # Get all ensemble directories
        sim_dirs = [d for d in self.base_dir.iterdir() if d.is_dir() and d.name.startswith('sim')]
        # Restrict to the subset of members used for training (1..MAX_ENSEMBLE_MEMBER; None = all)
        if MAX_ENSEMBLE_MEMBER is not None:
            sim_dirs = [d for d in sim_dirs if 1 <= int(d.name.replace('sim', '')) <= MAX_ENSEMBLE_MEMBER]
        sim_dirs.sort()  # Ensure they're in order
        n_ensembles = len(sim_dirs)
        logging.info(f"Found {n_ensembles} ensemble directories")
        
        # Get ensemble numbers from directory names
        ensemble_nums = [int(d.name.replace('sim', '')) for d in sim_dirs]
        # Try to load indices if file exists
        loaded_indices = False
        if indices_file and os.path.exists(indices_file):
            try:
                with open(indices_file, 'r') as f:
                    content = f.read().strip()
                    if content:  # Check if file is not empty
                        split_indices = json.loads(content)
                        self.ensemble_indices = np.array(split_indices[split])
                        loaded_indices = True
                        logging.info(f"Successfully loaded indices from {indices_file}")
            except (json.JSONDecodeError, KeyError) as e:
                logging.warning(f"Could not load indices from {indices_file}: {str(e)}")
                loaded_indices = False
        
        if not loaded_indices:
            # Generate new random indices
            logging.info("Generating new random split with 75/25 train/val")
            rng = np.random.RandomState(seed)
            indices = np.array(ensemble_nums)
            rng.shuffle(indices)
            
            # Split indices into train (75%) and validation (25%).
            # The test set is not split here: test members were held out and
            # handled separately during postprocessing.
            train_idx = int(0.75 * n_ensembles)
            
            split_indices = {
                'train': indices[:train_idx].tolist(),
                'val': indices[train_idx:].tolist()
            }
            
            # Save indices if filename is provided
            if indices_file:
                try:
                    logging.info(f"Saving indices to {indices_file}")
                    with open(indices_file, 'w') as f:
                        json.dump(split_indices, f)
                except IOError as e:
                    logging.error(f"Could not save indices to {indices_file}: {str(e)}")
            
            self.ensemble_indices = np.array(split_indices[split])
        
        # Create file paths list for each ensemble in this split
        for ensemble_num in self.ensemble_indices:
            sim_dir = f"sim{ensemble_num:03d}"
            file_path = self.base_dir / sim_dir / "processed_surface_data.nc"
            if file_path.exists():
                self.file_paths.append(file_path)
            else:
                logging.warning(f"File not found: {file_path}")
        
        logging.info(f"Dataset Info:")
        logging.info(f"- Split: {split}")
        logging.info(f"- Number of ensembles: {len(self.file_paths)}")
        logging.info(f"{'==='*20}")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        # Load data from netCDF file
        ds = xr.open_dataset(file_path)
        
        # Extract features
        soil_moisture_features = ds.soil_moisture.values  # Shape: [time, lat, lon]
        sensible_heat_features = ds.sensible_heat.values
        latent_heat_features = ds.latent_heat.values
        target_features = ds.smap.values
        
        # Close dataset
        ds.close()
        
        # Keep every timestep in the file (no temporal subsetting)
        soil_moisture = soil_moisture_features
        sensible_heat = sensible_heat_features
        latent_heat = latent_heat_features
        target = target_features
        
        # Stack features
        features = np.stack([soil_moisture, sensible_heat, latent_heat], axis=0)
        target = target[None, ...]  # Add channel dimension
        
        # Convert to tensor and handle NaNs
        features = torch.from_numpy(features).float()
        target = torch.from_numpy(target).float()
        features = torch.nan_to_num(features, 0)
        target = torch.nan_to_num(target, 0)
        
        return features, target

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('training.log')
        ]
    )

    logging.info(f"\n{'='*80}")
    logging.info("Starting Soil Moisture Prediction Training with 3D UNET")
    logging.info(f"{'='*80}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Data path to the ensemble directory (configured via DATA_DIR above)
    data_path = Path(DATA_DIR)
    logging.info(f"Loading data from: {data_path}")
    
    # Initialize model and training components
    model = SoilMoistureUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    criterion = torch.nn.L1Loss()

    # Create datasets and dataloaders
    train_dataset = SoilMoistureEnsembleDataset(data_path, split='train', seed=42, indices_file='ensemble_split_indices.json')
    val_dataset = SoilMoistureEnsembleDataset(data_path, split='val', indices_file='ensemble_split_indices.json')

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

    # Setup logging and saving directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path('runs/soil_moisture') / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # Training parameters
    num_epochs = 200
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0

    # Setup warmup scheduler
    base_lr = 1e-5
    warmup_epochs = 10
    scheduler = WarmupScheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=num_epochs,
        base_lr=base_lr,
        warmup_lr=1e-6
    )

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        for data, target in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_mae = 0
        val_mse = 0
        val_r2 = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                # Calculate metrics
                mae = torch.mean(torch.abs(output - target))
                mse = torch.mean((output - target) ** 2)
                rmse = torch.sqrt(mse)
                
                # For R2 score
                target_mean = torch.mean(target)
                ss_tot = torch.sum((target - target_mean) ** 2)
                ss_res = torch.sum((target - output) ** 2)
                r2 = 1 - (ss_res / ss_tot)
                
                # Accumulate metrics
                val_loss += criterion(output, target).item()
                val_mae += mae.item()
                val_mse += mse.item()
                val_r2 += r2.item()
                num_val_batches += 1

        # Calculate average losses and metrics
        train_loss /= len(train_loader)
        val_loss /= num_val_batches
        val_mae /= num_val_batches
        val_rmse = np.sqrt(val_mse / num_val_batches)
        val_r2 /= num_val_batches

        # Store losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Log progress
        logging.info(f'Epoch {epoch+1}/{num_epochs}:')
        logging.info(f'Training Loss: {train_loss:.4f}')
        logging.info(f'Validation Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, R2: {val_r2:.4f}')

        # Log to tensorboard
        writer.add_scalars('Loss/epoch', {
            'train': train_loss,
            'val': val_loss
        }, epoch)

        writer.add_scalars('Metrics/epoch', {
            'mae': val_mae,
            'rmse': val_rmse,
            'r2': val_r2
        }, epoch)

        # Update learning rate
        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_rate', current_lr, epoch)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = Path(log_dir) / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': best_val_loss,
            }, save_path)
            logging.info(f"Saved best model with val_loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_path = Path(log_dir) / f'checkpoint_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, save_path)
            logging.info(f"Saved checkpoint at epoch {epoch+1}")

    # Save final losses
    np.save(Path(log_dir) / 'train_losses.npy', train_losses)
    np.save(Path(log_dir) / 'val_losses.npy', val_losses)

    writer.close()
    logging.info("Training completed successfully!")

if __name__ == '__main__':
    main()
