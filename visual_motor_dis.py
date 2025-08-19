# import os
# import math
# import h5py
# import numpy as np
# import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_motor_data(data_dir="/home/jolia/vr-hand-tracking/Franka-Teach/RUKA/data/left_hand/demonstration_left"):
    """Load motor position data"""
    data_path = Path(data_dir)
    
    # Load motor positions
    present_positions = np.load(data_path / "present_positions.npy")
    
    # Load timestamps if available
    try:
        timestamps = np.load(data_path / "timestamps.npy")
    except:
        timestamps = np.arange(len(present_positions))
    
    return present_positions, timestamps

def get_finger_motor_mapping():
    """Get mapping of finger names to motor IDs"""
    FINGER_NAMES_TO_MOTOR_IDS = {
        "Index": [3, 4],
        "Middle": [5, 6],
        "Ring": [8, 7],
        "Pinky": [10, 9],     # DIP, MCP (note: Pinky MCP is motor 8)
        "Thumb": [0, 1, 2]    # DIP, PIP, MCP
    }
    
    return FINGER_NAMES_TO_MOTOR_IDS

def create_motor_distribution_plots(data_dir="/home/jolia/vr-hand-tracking/Franka-Teach/RUKA/data/right_hand/demonstration_right_four_fingers"):
    """Create plots showing present motor position distributions with statistics"""
    
    print(f"Loading motor data from {data_dir}...")
    present_positions, timestamps = load_motor_data(data_dir)
    finger_motor_ids = get_finger_motor_mapping()
    
    print(f"Data shape: {present_positions.shape}")
    print(f"Time range: {timestamps[0]:.2f} to {timestamps[-1]:.2f} seconds")
    
    # Filter out thumb and create separate plots for DIP and MCP
    non_thumb_fingers = {k: v for k, v in finger_motor_ids.items() if k != "Thumb"}
    
    # Create subplots: 4 fingers x 2 motors (DIP and MCP) = 8 subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Present Motor Position Distributions - {data_dir}', fontsize=16, fontweight='bold')
    
    colors = ['orange', 'blue']
    
    plot_idx = 0
    for finger_name, motor_ids in non_thumb_fingers.items():
        # Extract motor data for this finger
        finger_present = present_positions[:, motor_ids]
        
        # Create motor labels (DIP and MCP for non-thumb fingers)
        motor_labels = ["DIP", "MCP"]
        
        # Create separate histogram for each motor (DIP and MCP)
        for i, motor_id in enumerate(motor_ids):
            row = plot_idx // 4
            col = plot_idx % 4
            ax = axes[row, col]
            
            motor_data = finger_present[:, i]
            
            # Calculate statistics
            mean_val = motor_data.mean()
            std_val = motor_data.std()
            min_val = motor_data.min()
            max_val = motor_data.max()
            
            # Create histogram
            ax.hist(motor_data, bins=50, alpha=0.7, color=colors[i], density=True)
            
            # Add vertical lines for statistics
            ax.axvline(mean_val, color=colors[i], linestyle='-', linewidth=2, 
                      label=f"Mean: {mean_val:.0f}")
            ax.axvline(mean_val + std_val, color=colors[i], linestyle='--', linewidth=1,
                      label=f"Mean+Std: {mean_val + std_val:.0f}")
            ax.axvline(mean_val - std_val, color=colors[i], linestyle='--', linewidth=1,
                      label=f"Mean-Std: {mean_val - std_val:.0f}")
            ax.axvline(min_val, color=colors[i], linestyle=':', linewidth=1,
                      label=f"Min: {min_val:.0f}")
            ax.axvline(max_val, color=colors[i], linestyle=':', linewidth=1,
                      label=f"Max: {max_val:.0f}")
            
            # Customize plot
            ax.set_xlabel('Motor Position')
            ax.set_ylabel('Density')
            ax.set_title(f'{finger_name} - {motor_labels[i]} (Motor {motor_id})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add motor statistics
            stats_text = f"Min: {min_val:.0f}\n"
            stats_text += f"Max: {max_val:.0f}\n"
            stats_text += f"Mean: {mean_val:.0f}\n"
            stats_text += f"Std: {std_val:.0f}"
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    verticalalignment='top', fontsize=9)
            
            plot_idx += 1
    
    plt.tight_layout()
    plt.savefig(f'present_motor_distributions_{data_dir.replace("/", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return present_positions, timestamps

def print_motor_statistics(present_positions, data_dir="left_hand/demonstration_1"):
    """Print statistics for present motor data"""
    
    finger_motor_ids = get_finger_motor_mapping()
    
    print(f"\n=== Present Motor Statistics for {data_dir} ===")
    print("-" * 80)
    
    for finger_name, motor_ids in finger_motor_ids.items():
        finger_present = present_positions[:, motor_ids]
        
        # Create motor labels
        if finger_name == "Thumb":
            motor_labels = ["DIP", "PIP", "MCP"]
        else:
            motor_labels = ["DIP", "MCP"]
        
        print(f"\n{finger_name} Finger:")
        print(f"{'Motor':<8} {'Min':<8} {'Max':<8} {'Mean':<8} {'Std':<8} {'Mean-Std':<10} {'Mean+Std':<10}")
        print("-" * 70)
        
        for i, motor_id in enumerate(motor_ids):
            motor_data = finger_present[:, i]
            mean_val = motor_data.mean()
            std_val = motor_data.std()
            min_val = motor_data.min()
            max_val = motor_data.max()
            
            print(f"{motor_labels[i]:<8} {min_val:<8.0f} {max_val:<8.0f} {mean_val:<8.1f} {std_val:<8.1f} "
                  f"{mean_val-std_val:<10.1f} {mean_val+std_val:<10.1f}")

def create_box_plots(present_positions, data_dir="left_hand/demonstration_1"):
    """Create box plots showing motor position distributions"""
    
    finger_motor_ids = get_finger_motor_mapping()
    
    # Create subplots for each finger
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Motor Position Box Plots - {data_dir}', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    for idx, (finger_name, motor_ids) in enumerate(finger_motor_ids.items()):
        ax = axes[idx]
        
        # Extract motor data for this finger
        finger_present = present_positions[:, motor_ids]
        
        # Create motor labels
        if finger_name == "Thumb":
            motor_labels = ["DIP", "PIP", "MCP"]
        else:
            motor_labels = ["DIP", "MCP"]
        
        # Create box plot
        box_data = [finger_present[:, i] for i in range(len(motor_ids))]
        bp = ax.boxplot(box_data, labels=motor_labels, patch_artist=True)
        
        # Color the boxes
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors[:len(motor_ids)]):
            patch.set_facecolor(color)
        
        # Customize plot
        ax.set_ylabel('Motor Position')
        ax.set_title(f'{finger_name} Finger')
        ax.grid(True, alpha=0.3)
        
        # Add individual motor statistics
        stats_text = ""
        for i, motor_id in enumerate(motor_ids):
            motor_data = finger_present[:, i]
            mean_val = motor_data.mean()
            std_val = motor_data.std()
            min_val = motor_data.min()
            max_val = motor_data.max()
            
            stats_text += f"{motor_labels[i]} (M{motor_id}):\n"
            stats_text += f"  Min: {min_val:.0f}\n"
            stats_text += f"  Max: {max_val:.0f}\n"
            stats_text += f"  Mean: {mean_val:.0f}\n"
            stats_text += f"  Std: {std_val:.0f}\n\n"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top', fontsize=8)
    
    # Remove the last subplot if we have fewer fingers than subplots
    if len(finger_motor_ids) < 6:
        axes[-1].remove()
    
    plt.tight_layout()
    plt.savefig(f'motor_box_plots_{data_dir.replace("/", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run the analysis"""
    
    data_dir = "/home/jolia/vr-hand-tracking/Franka-Teach/RUKA/data/right_hand/demonstration_right_thumb"
    
    try:
        # Create distribution plots
        print("Creating present motor distribution plots...")
        present_positions, timestamps = create_motor_distribution_plots(data_dir)
        
        # # Create box plots
        # print("Creating motor box plots...")
        # create_box_plots(present_positions, data_dir)
        
        # # Print statistics
        # print("Calculating motor statistics...")
        # print_motor_statistics(present_positions, data_dir)
        
        print(f"\nAnalysis complete! Check the generated PNG files:")
        print(f"  - present_motor_distributions_{data_dir.replace('/', '_')}.png")
        print(f"  - motor_box_plots_{data_dir.replace('/', '_')}.png")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 