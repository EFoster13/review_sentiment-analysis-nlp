import os
import shutil

def create_folder(path):
    """Create folder if it doesn't exist"""
    os.makedirs(path, exist_ok=True)
    print(f"✓ Created: {path}")

def move_file(src, dest):
    """Move file to new location"""
    if os.path.exists(src):
        shutil.move(src, dest)
        print(f"✓ Moved: {src} → {dest}")

print("="*60)
print("ORGANIZING PROJECT")
print("="*60)

# Create organized structure
print("\nCreating organized folders...")
create_folder("data")
create_folder("data/cleaned")
create_folder("models")
create_folder("scripts")
create_folder("logs")

# Move cleaned datasets
print("\nMoving datasets...")
move_file("combined_dataset.csv", "data/cleaned/combined_dataset.csv")
move_file("imdb_train_cleaned.csv", "data/cleaned/imdb_train_cleaned.csv")
move_file("imdb_test_cleaned.csv", "data/cleaned/imdb_test_cleaned.csv")
move_file("yelp_reviews_cleaned.csv", "data/cleaned/yelp_reviews_cleaned.csv")

# Move model
print("\nMoving model...")
if os.path.exists("final_model"):
    move_file("final_model", "models/final_model")

# Move scripts
print("\nMoving scripts...")
scripts = [
    "preprocess_data.py",
    "clean_yelp.py",
    "combine_datasets.py",
    "train_bert.py",
    "train_combined_model.py",
    "test_model.py",
]

for script in scripts:
    if os.path.exists(script):
        move_file(script, f"scripts/{script}")

print("\n" + "="*60)
print("ORGANIZATION COMPLETE!")
print("="*60)
print("\nNew structure:")
print("""
NLP_Project/
├── data/
│   └── cleaned/          # All cleaned datasets
├── models/
│   └── final_model/      # Your trained model
├── scripts/              # Training/processing scripts
├── logs/                 # Future log files
├── test_cross_domain.py  # Testing script (stays in root)
└── cleanup_project.py    # This script
""")