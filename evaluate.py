import os
import argparse
import torch
import torchvision as tv
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import utils.misc as misc
from model.networks import Generator
from utils.data import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default="configs/finetune_mit.yaml", 
                    help="Path to yaml config file")
parser.add_argument('--checkpoint', type=str, default="", 
                    help="Path to checkpoint file")
parser.add_argument('--output_dir', type=str, default="results", 
                    help="Directory to save results")
parser.add_argument('--num_samples', type=int, default=10, 
                    help="Number of samples to evaluate")


def evaluate_model(generator, test_dataloader, config, args):
    """
    Evaluate the model on the test dataset
    """
    device = torch.device('cuda' if torch.cuda.is_available() 
                         and config.use_cuda_if_available else 'cpu')
    
    generator.eval()
    
    # Metrics
    psnr_scores = []
    ssim_scores = []
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, batch_real in enumerate(tqdm(test_dataloader)):
            if i >= args.num_samples:
                break
                
            batch_real = batch_real.to(device)
            
            # Create mask
            bbox = misc.random_bbox(config)
            regular_mask = misc.bbox2mask(config, bbox).to(device)
            irregular_mask = misc.brush_stroke_mask(config).to(device)
            mask = torch.logical_or(irregular_mask, regular_mask).to(torch.float32)
            
            # Prepare input
            batch_incomplete = batch_real*(1.-mask)
            ones_x = torch.ones_like(batch_incomplete)[:, 0:1].to(device)
            x = torch.cat([batch_incomplete, ones_x, ones_x*mask], axis=1)
            
            # Generate inpainted images
            x1, x2 = generator(x, mask)
            
            # Apply mask to get completed image
            batch_complete = x2*mask + batch_incomplete*(1.-mask)
            
            # Convert to numpy arrays for metrics calculation
            real_np = batch_real.cpu().permute(0, 2, 3, 1).numpy() * 0.5 + 0.5
            comp_np = batch_complete.cpu().permute(0, 2, 3, 1).numpy() * 0.5 + 0.5
            mask_np = mask.cpu().permute(0, 2, 3, 1).numpy()
            
            # Calculate metrics for each image in batch
            for b in range(batch_real.size(0)):
                # Calculate PSNR
                masked_region = mask_np[b, :, :, 0] > 0.5
                if np.sum(masked_region) > 0:  # Only if mask is not empty
                    # Calculate PSNR only on the masked region
                    real_masked = real_np[b, masked_region]
                    comp_masked = comp_np[b, masked_region]
                    psnr_val = psnr(real_masked, comp_masked, data_range=1.0)
                    psnr_scores.append(psnr_val)
                
                    # Calculate SSIM on the whole image
                    ssim_val = ssim(
                        real_np[b], comp_np[b], 
                        multichannel=True, 
                        channel_axis=2, 
                        data_range=1.0
                    )
                    ssim_scores.append(ssim_val)
            
            # Save visualization
            viz_images = [
                misc.pt_to_image(batch_real),
                misc.pt_to_image(batch_incomplete),
                misc.pt_to_image(batch_complete)
            ]
            
            # Create a grid of images
            img_grid = tv.utils.make_grid(
                torch.cat([img[:min(4, img.size(0))] for img in viz_images]), 
                nrow=4
            )
            
            # Save the grid
            tv.utils.save_image(img_grid, f"{args.output_dir}/sample_{i}.png")
    
    # Print metrics
    print(f"Average PSNR: {np.mean(psnr_scores):.4f} ± {np.std(psnr_scores):.4f}")
    print(f"Average SSIM: {np.mean(ssim_scores):.4f} ± {np.std(ssim_scores):.4f}")
    
    # Save metrics to file
    with open(f"{args.output_dir}/metrics.txt", "w") as f:
        f.write(f"Average PSNR: {np.mean(psnr_scores):.4f} ± {np.std(psnr_scores):.4f}\n")
        f.write(f"Average SSIM: {np.mean(ssim_scores):.4f} ± {np.std(ssim_scores):.4f}\n")


def main():
    args = parser.parse_args()
    config = misc.get_config(args.config)
    
    # If checkpoint is not provided, use the one from config
    if args.checkpoint == "":
        args.checkpoint = os.path.join(config.checkpoint_dir, "states.pth")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() 
                         and config.use_cuda_if_available else 'cpu')
    
    # Load test dataset
    test_dataset = ImageDataset(
        config.dataset_path.replace('train', 'test'),
        img_shape=config.img_shapes,
        random_crop=config.random_crop,
        scan_subdirs=config.scan_subdirs
    )
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # Construct network
    cnum_in = config.img_shapes[2]
    generator = Generator(cnum_in=cnum_in+2, cnum_out=cnum_in, cnum=48, return_flow=False)
    generator = generator.to(device)
    
    # Load checkpoint
    state_dicts = torch.load(args.checkpoint, map_location=device)
    generator.load_state_dict(state_dicts['G'])
    print(f"Loaded model from: {args.checkpoint}")
    
    # Evaluate
    evaluate_model(generator, test_dataloader, config, args)


if __name__ == "__main__":
    main() 