#!/usr/bin/env python3
"""
LeRobotDataset to MaskImageDataset Converter

This script converts LeRobotDataset format to MaskImageDataset format used in DexGraspVLA.
Uses the official LeRobotDataset API for loading and processing datasets.

Author: AI Assistant
Date: 2025
"""

import os
import json
import argparse
import numpy as np
from requests import head
import zarr
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor

# Computer vision model imports
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model
from segment_anything import sam_model_registry, SamPredictor

# LeRobot imports
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from controller.common.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
import time
import yaml
from inference_utils.utils import log,show_mask


class LeRobotDatasetConverter:
    """Convert LeRobotDataset to MaskImageDataset format using official LeRobotDataset API"""
    
    def __init__(self, 
                 dataset_id: str,
                 output_path: str,
                 image_size: Tuple[int, int] = (518, 518),
                 max_episodes: Optional[int] = None,
                 val_ratio: float = 0.1,
                 root: Optional[str] = None):
        """
        Initialize the converter
        
        Args:
            dataset_id: LeRobot dataset ID (e.g., "lerobot/aloha_sim_insertion_human")
            output_path: Output path for Zarr dataset
            image_size: Target image size (H, W)
            max_episodes: Maximum number of episodes to convert
            val_ratio: Validation set ratio
            root: Local dataset root path (if dataset is local)
        """
        self.dataset_id = dataset_id
        self.output_path = Path(output_path)
        self.image_size = image_size
        self.max_episodes = max_episodes
        self.val_ratio = val_ratio
        self.root = root
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.config = self.load_inference_config()
        self.img_dir = os.path.join(output_path, 'images')
        os.makedirs(self.img_dir, exist_ok=True)
        self.log_file_path = os.path.join(output_path, 'run.log')
        self.log_file = open(self.log_file_path, 'w')
        
        # Initialize dataset
        self.dataset = LeRobotDataset(self.dataset_id, root=self.root,local_files_only=True)

        self.init_controller()

    def init_controller(self):
        self.device = torch.device('cuda:0')

        # main_config_path = os.path.join(os.path.dirname(__file__), 'controller', 'config', 'train_dexgraspvla_controller_workspace.yaml')
        # task_config_path = os.path.join(os.path.dirname(__file__), 'controller', 'config', 'task', 'grasp.yaml')
        
        # self.cfg = load_config(
        #     main_config_path=main_config_path,
        #     task_config_path=task_config_path
        # )
        # workspace = hydra.utils.get_class(self.cfg._target_)(self.cfg)
        # self.policy = workspace.model
        # self.policy.eval().to(self.device)

        # Initialize SAM
        sam_checkpoint = self.config['sam']['checkpoint']
        model_type = self.config['sam']['model_type']
        print(f'model type:{model_type}, sam checkpoint: {sam_checkpoint}')
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
        
        # Initialize Cutie
        self.cutie = get_default_model()
        self.processor = InferenceCore(self.cutie, cfg=self.cutie.cfg)
        self.processor.max_internal_size = -1

    def mark_bbox_manual(self, third_color_image):
        # Save original image
        self.save_image(third_color_image, "head_image_start")
        self.log("Head camera image saved at the beginning of the episode.", message_type="info")
        # Display image and get bounding box
        plt.figure()
        plt.imshow(third_color_image)  # The image is BGR
        plt.axis('off')
        plt.title("Please click two points to define the bounding box (top left and bottom right)")
        bbox_points = plt.ginput(n=2, timeout=0)
        plt.close()
        (x1, y1), (x2, y2) = bbox_points
        bbox = np.array([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])

        # Save image with bounding box
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        img_with_bbox_filename = f"{timestamp}_head_image_with_bbox.png"
        self.show_and_save_image_with_bbox(third_color_image, bbox, img_with_bbox_filename)
        return bbox

    def initialize_sam_cutie(self, third_color_image, bbox):
        self.processor.clear_memory()
        torch.cuda.empty_cache()
        self.predictor.set_image(third_color_image)

        masks, scores, _ = self.predictor.predict(box=bbox, multimask_output=True)
        # cv2.imwrite('mask1.png', masks[0].astype(np.uint8)*255)
        self.best_mask = masks[np.argmax(scores)]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        img_with_mask_filename = f"{timestamp}_head_image_with_mask.png"
        self.show_and_save_image_with_mask(third_color_image, self.best_mask, img_with_mask_filename)

        # Reinitialize Cutie
        self.mask = torch.from_numpy(self.best_mask.astype('uint8')).cuda()
        self.objects = np.unique(self.best_mask.astype('uint8'))
        self.objects = self.objects[self.objects != 0].tolist()
        self.cutie_initialized = False 

    

    def show_and_save_image_with_bbox(self, image, bbox, filename):
        """Save and display image with bounding box
        
        Args:
            image: Original image
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            filename: Filename to save
        """
        # Create an image with the same size as the original
        height, width = image.shape[:2]
        fig = plt.figure(figsize=(width/100, height/100), dpi=100)
        ax = plt.Axes(fig, [0., 0., 1., 1.])  # Create axes without margins
        ax.set_axis_off()
        fig.add_axes(ax)
        
        # Display image
        ax.imshow(image)
        
        # Get bounding box color and line width from config
        bbox_color = self.config['visualization']['bbox']['color']
        bbox_linewidth = self.config['visualization']['bbox']['linewidth']
        
        # Draw bounding box
        x1, y1, x2, y2 = bbox.astype(int)
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                            linewidth=bbox_linewidth, 
                            edgecolor=bbox_color, 
                            facecolor='none')
        ax.add_patch(rect)
        
        # Save image
        img_path = os.path.join(self.img_dir, filename)
        plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
        
        # Display image
        plt.draw()
        plt.pause(0.5)
        plt.close(fig)
        
        self.log(f"Head camera image with bounding box saved.", message_type="info")

    def show_and_save_image_with_mask(self, image, mask, filename):
        """Save and display image with mask
        
        Args:
            image: Original image
            mask: Binary mask
            filename: Filename to save
        """
        # Create an image with the same size as the original
        height, width = image.shape[:2]
        fig = plt.figure(figsize=(width/100, height/100), dpi=100)
        ax = plt.Axes(fig, [0., 0., 1., 1.])  # Create axes without margins
        ax.set_axis_off()
        fig.add_axes(ax)
        
        # Display image
        ax.imshow(image)
        
        # Get mask color and random color settings from config
        mask_color = self.config['visualization']['mask']['color']
        
        # Display mask with configured color
        show_mask(mask, ax, color=mask_color)
        
        # Save image
        img_path = os.path.join(self.img_dir, filename)
        plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
        
        # Display image
        plt.draw()
        plt.pause(0.5)
        plt.close(fig)
        
        self.log(f"Head camera image with mask saved.", message_type="info")

    def save_image(self, image, filename):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        img_path = os.path.join(self.img_dir, f"{timestamp}_{filename}.png")
        cv2.imwrite(img_path, image)

    def log(self, message, message_type = None):
        log(message, message_type, self.log_file)

    def load_inference_config(self):
        """Load system configuration from YAML file"""
        config_path = os.path.join(os.path.dirname(__file__), 'inference_utils', 'config.yaml')
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)


    
    
    def _get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information and metadata"""
        info = {
            'dataset_id': self.dataset_id,
            'total_items': len(self.dataset),
            # 'episode_ends': self.dataset.episode_ends,
            # 'camera_keys': self.dataset.camera_keys,
            # 'video': self.dataset.video,
            # 'encoding': self.dataset.encoding
        }
        
        # Get sample structure
        if len(self.dataset) > 0:
            sample = self.dataset[0]
            info['sample_keys'] = list(sample.keys())
            if 'observation' in sample:
                obs = sample['observation']
                info['observation_keys'] = list(obs.keys())
                if 'images' in obs:
                    images = obs['images']
                    info['image_keys'] = list(images.keys())
                    
        return info
    
    def _extract_video_frames(self, video_path: str, frame_indices: List[int]) -> np.ndarray:
        """Extract frames from MP4 video"""
        if not os.path.exists(video_path):
            print(f"Warning: Video file not found: {video_path}")
            return np.zeros((len(frame_indices), 480, 640, 3), dtype=np.uint8)
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            else:
                # If frame not found, create black frame
                height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                if height == 0 or width == 0:
                    height, width = 480, 640
                frames.append(np.zeros((height, width, 3), dtype=np.uint8))
        
        cap.release()
        return np.array(frames)
    
    def _process_image_batch(self, images: np.ndarray) -> np.ndarray:
        """Process images to target size and format"""
        if len(images.shape) == 3:
            images = images[None, ...]  # Add batch dimension
        
        # Convert to torch tensor
        images_tensor = torch.from_numpy(images).float()
        
        # Normalize to [0, 1]
        images_tensor = images_tensor / 255.0
        
        # Resize to target size
        images_tensor = F.interpolate(
            images_tensor.permute(0, 3, 1, 2),  # [B, C, H, W]
            size=self.image_size,
            mode='bilinear',
            align_corners=False
        )
        
        return images_tensor.permute(0, 2, 3, 1).numpy()  # [B, H, W, C]
    
    def _create_mask_from_rgb(self, rgb_image: np.ndarray) -> np.ndarray:
        """Create a simple mask from RGB image (placeholder implementation)"""
        # This is a placeholder - in real implementation, you might want to:
        # 1. Use segmentation models
        # 2. Use depth information
        # 3. Use pre-computed masks from LeRobotDataset
        
        # Simple threshold-based mask (for demonstration)
        #print(type(rgb_image))
        with torch.no_grad():
            image_tensor = to_tensor(rgb_image.copy()).cuda().float()
            if self.cutie_initialized == False:
                output_prob = self.processor.step(image_tensor, self.mask, objects=self.objects)
                self.cutie_initialized = True
            else:
                output_prob = self.processor.step(image_tensor)
            current_mask = self.processor.output_prob_to_mask(output_prob)
            current_mask_np = current_mask.cpu().numpy().astype(np.uint8)
        
        # gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        # _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        #cv2.imwrite('mask.png', current_mask_np)
        #mask = mask.astype(np.float32) / 255.0
        
        return current_mask_np # Add channel dimension
    
    def _combine_rgb_and_mask(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Combine RGB and mask into RGBM format"""
        # timestamp = time.strftime("%Y%m%d_%H%M%S")
        # img_with_mask_filename = f"{timestamp}_head_image_with_mask.png"
        # self.show_and_save_image_with_mask(rgb, mask, img_with_mask_filename)
        return np.concatenate([rgb, mask[..., None]], axis=-1)
    
    def _extract_episode_data(self, from_id: int, to_id: int) -> Dict[str, np.ndarray]:
        """Extract data from a single episode using LeRobot API"""
        actions = []
        states = []
        right_cam_images = []
        rgbm_images = []
        
        # Get episode length
        #episode_length = len(episode_data.get('action', []))

        for idx in range(from_id, to_id):
            episode_data = self.dataset[idx]
            # Extract action and state
            if 'action' in episode_data:
                actions.append(episode_data['action'][7:14])
            
            if 'observation.state' in episode_data:
                states.append(episode_data['observation.state'][7:14])
                #print(episode_data['observation.state'])
            
            # Extract images
            if 'observation.images.front' in episode_data:
                
                # Create mask (placeholder implementation)
                head_img = episode_data['observation.images.front'].permute(1, 2, 0).numpy()
                if head_img.max() <= 1.0:
                    head_img = (head_img * 255).astype(np.uint8)
                if idx == from_id:
                    bbox = self.mark_bbox_manual(head_img)
                    self.initialize_sam_cutie(head_img, bbox)
                mask = self._create_mask_from_rgb(head_img)
                rgbm_img = self._combine_rgb_and_mask(head_img, mask)
                rgbm_images.append(rgbm_img)
            if 'observation.images.wrist_right' in episode_data:
                right_img = episode_data['observation.images.wrist_right'].permute(1, 2, 0).numpy()
                if right_img.max() <= 1.0:
                    right_img = (right_img * 255).astype(np.uint8)
                # cv2.imwrite('right_img.png', right_img)
                right_cam_images.append(right_img)
                        
        return {
            'action': np.array(actions, dtype=np.float32),
            'right_state': np.array(states, dtype=np.float32),
            'right_cam_img': np.array(right_cam_images, dtype=np.uint8),
            'rgbm': np.array(rgbm_images, dtype=np.uint8)
        }
    
    def _get_episode_boundaries(self) -> List[Tuple[int, int]]:
        """Get episode boundaries from dataset"""
        episode_ends = self.dataset.episode_ends
        boundaries = []
        start_idx = 0
        for end_idx in episode_ends:
            boundaries.append((start_idx, end_idx))
            start_idx = end_idx
        return boundaries
    
    def convert(self):
        """Convert LeRobotDataset to MaskImageDataset format using ReplayBuffer"""
        print(f"Converting LeRobotDataset: {self.dataset_id}")
        
        # Get dataset info
        dataset_info = self._get_dataset_info()
        print(f"Dataset info: {dataset_info}")
        
        # Get episode boundaries
        #episode_boundaries = self._get_episode_boundaries()
        n_episodes = self.dataset.num_episodes
        
        # if self.max_episodes:
        #     n_episodes = min(n_episodes, self.max_episodes)
        #     episode_boundaries = episode_boundaries[:n_episodes]
        
        print(f"Converting {n_episodes} episodes...")
        
        # Create ReplayBuffer-backed Zarr store
        buffer = ReplayBuffer.create_empty_zarr(storage=zarr.DirectoryStore(str(self.output_path)))
        
        total_samples = 0
        for episode_idx in range(n_episodes):
            print(f"Processing episode {episode_idx + 1}/{n_episodes}")
            try:
                # Extract episode data using LeRobotDataset API
                episode_from_idx = self.dataset.episode_data_index['from'][episode_idx]
                episode_to_idx = self.dataset.episode_data_index['to'][episode_idx]
                # episode_data = self.dataset[episode_from_idx:episode_to_idx]
                processed_data = self._extract_episode_data(episode_from_idx, episode_to_idx)
                if len(processed_data['action']) == 0:
                    print(f"Warning: Episode {episode_idx} has no action data, skipping")
                    continue
                # 插入到ReplayBuffer
                buffer.add_episode(processed_data)
                total_samples += len(processed_data['action'])
            except Exception as e:
                print(f"Error processing episode {episode_idx}: {e}")
                continue
        if buffer.n_episodes == 0:
            raise ValueError("No valid episodes found!")
        # 保存metadata.json
        metadata = {
            'dataset_id': self.dataset_id,
            'n_episodes': buffer.n_episodes,
            'total_samples': total_samples,
            'image_size': self.image_size,
            'val_ratio': self.val_ratio,
            'action_dim': buffer.data['action'].shape[1],
            'state_dim': buffer.data['right_state'].shape[1],
            'dataset_info': dataset_info
        }
        with open(self.output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Conversion completed! Output saved to {self.output_path}")
        print(f"Total episodes: {buffer.n_episodes}")
        print(f"Total samples: {total_samples}")
        print(f"Image size: {self.image_size}")
        print(f"Action dimension: {buffer.data['action'].shape[1]}")
        print(f"State dimension: {buffer.data['right_state'].shape[1]}")
        return self.output_path


def main():
    parser = argparse.ArgumentParser(description='Convert LeRobotDataset to MaskImageDataset format')
    parser.add_argument('--dataset-id', type=str, required=True, 
                       help='LeRobot dataset ID (e.g., lerobot/aloha_sim_insertion_human)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for Zarr dataset')
    parser.add_argument('--image-size', type=int, nargs=2, default=[518, 518],
                       help='Target image size (H W)')
    parser.add_argument('--max-episodes', type=int, default=None,
                       help='Maximum number of episodes to convert')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='Validation set ratio')
    parser.add_argument('--root', type=str, default=None,
                       help='Local dataset root path (if dataset is local)')
    
    args = parser.parse_args()
    
    # Create converter and run conversion
    converter = LeRobotDatasetConverter(
        dataset_id=args.dataset_id,
        output_path=args.output,
        image_size=tuple(args.image_size),
        max_episodes=args.max_episodes,
        val_ratio=args.val_ratio,
        root=args.root
    )
    
    converter.convert()


if __name__ == "__main__":
    main() 