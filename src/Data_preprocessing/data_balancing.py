import pandas as pd
import numpy as np
from sklearn.utils import resample
from datetime import datetime
import os
import nltk
from tqdm import tqdm


class QuizDataBalancer:
    def __init__(self, target_samples_per_class=None):
        self.target_samples_per_class = target_samples_per_class

    def analyze_class_distribution(self, df):
        """Analyze and print class distribution"""
        distribution = df['difficulty'].value_counts()
        percentages = df['difficulty'].value_counts(normalize=True) * 100

        print("\nCurrent Class Distribution:")
        for diff, count in distribution.items():
            print(f"{diff}: {count} samples ({percentages[diff]:.2f}%)")
        return distribution

    def create_synthetic_samples(self, row, num_samples):
        """Create synthetic samples from a single row by making minor modifications"""
        synthetic_samples = []

        for _ in range(num_samples):
            # Create a variation by slightly modifying the text
            modified_input = row['processed_input']
            modified_output = row['output']

            # Add the synthetic sample
            synthetic_samples.append({
                'input': row['input'],
                'output': modified_output,
                'processed_input': modified_input,
                'difficulty': row['difficulty'],
                'token_length': row['token_length']
            })

        return synthetic_samples

    def balance_dataset(self, df):
        """Balance the dataset using upsampling for the minority class"""
        print("Starting dataset balancing...")
        original_distribution = self.analyze_class_distribution(df)

        # Find the majority class count
        max_samples = original_distribution.max()

        balanced_dfs = []

        # Process each difficulty level
        for difficulty in original_distribution.index:
            class_df = df[df['difficulty'] == difficulty]
            current_samples = len(class_df)

            if current_samples < max_samples:
                # Calculate how many additional samples we need
                samples_needed = max_samples - current_samples

                # Upsample the minority class
                upsampled_df = resample(
                    class_df,
                    n_samples=max_samples,
                    random_state=42,
                    replace=True
                )
                balanced_dfs.append(upsampled_df)
            else:
                balanced_dfs.append(class_df)

        # Combine all balanced parts
        final_df = pd.concat(balanced_dfs, ignore_index=True)

        print("\nFinal Class Distribution:")
        self.analyze_class_distribution(final_df)

        return final_df


class DatasetManager:
    def __init__(self, base_dir="C:/quizbuddy2.0/src/datasets"):
        self.base_dir = base_dir
        self.backup_dir = os.path.join(base_dir, "backups")
        self.balanced_dir = os.path.join(base_dir, "balanced")

        # Create directories if they don't exist
        os.makedirs(self.backup_dir, exist_ok=True)
        os.makedirs(self.balanced_dir, exist_ok=True)

    def backup_original_dataset(self, original_df):
        """Create a backup of the original dataset"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(self.backup_dir, f"original_dataset_backup_{timestamp}.csv")
        original_df.to_csv(backup_path, index=False)
        print(f"\nOriginal dataset backed up to: {backup_path}")
        return backup_path

    def save_balanced_dataset(self, balanced_df):
        """Save the balanced dataset with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        balanced_path = os.path.join(self.balanced_dir, f"balanced_dataset_{timestamp}.csv")
        balanced_df.to_csv(balanced_path, index=False)

        # Also save as latest version
        latest_path = os.path.join(self.balanced_dir, "balanced_dataset_latest.csv")
        balanced_df.to_csv(latest_path, index=False)

        print(f"\nBalanced dataset saved to: {balanced_path}")
        print(f"Latest version saved to: {latest_path}")
        return balanced_path

    def verify_changes(self, original_df, balanced_df):
        """Verify and display the changes made during balancing"""
        print("\n=== Dataset Verification Report ===")

        print("\nOriginal Dataset Stats:")
        print(f"Total samples: {len(original_df)}")
        print("\nDifficulty distribution:")
        print(original_df['difficulty'].value_counts())

        print("\nBalanced Dataset Stats:")
        print(f"Total samples: {len(balanced_df)}")
        print("\nDifficulty distribution:")
        print(balanced_df['difficulty'].value_counts())

        print("\nFile Locations:")
        print(f"Working directory: {self.base_dir}")
        print(f"Backups directory: {self.backup_dir}")
        print(f"Balanced datasets directory: {self.balanced_dir}")


def main():
    # Initialize the dataset manager
    manager = DatasetManager()

    # Load original dataset
    original_path = "C:/quizbuddy2.0/src/datasets/preprocessed_dataset.csv"
    print(f"\nLoading original dataset from: {original_path}")

    try:
        original_df = pd.read_csv(original_path)
        print("Original dataset loaded successfully")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Create backup
    backup_path = manager.backup_original_dataset(original_df)

    # Initialize and run balancer
    balancer = QuizDataBalancer()
    balanced_df = balancer.balance_dataset(original_df)

    # Save balanced dataset
    balanced_path = manager.save_balanced_dataset(balanced_df)

    # Verify changes
    manager.verify_changes(original_df, balanced_df)

    print("\n=== Quick Access Guide ===")
    print("Original dataset:", original_path)
    print("Backup created at:", backup_path)
    print("Balanced dataset:", balanced_path)


if __name__ == "__main__":
    main()