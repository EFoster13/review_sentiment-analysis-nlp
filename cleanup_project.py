import shutil
import os

def delete_folder(path):
    """Safely delete a folder"""
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"✓ Deleted: {path}")
    else:
        print(f"⊘ Not found: {path}")

def delete_file(path):
    """Safely delete a file"""
    if os.path.exists(path):
        os.remove(path)
        print(f"✓ Deleted: {path}")
    else:
        print(f"⊘ Not found: {path}")

print("="*60)
print("CLEANING UP PROJECT")
print("="*60)

# Delete large raw data
print("\n1. Removing large raw data files...")
delete_folder("Yelp-JSON")
delete_folder("yelp_data")

# Delete old training checkpoints
print("\n2. Removing old training checkpoints...")
delete_folder("results")
delete_folder("results_combined")

# Delete unprocessed data
print("\n3. Removing unprocessed data files...")
delete_file("imdb_train.csv")
delete_file("imdb_test.csv")
delete_file("yelp_reviews.csv")

# Delete temporary scripts
print("\n4. Removing temporary scripts...")
scripts_to_delete = [
    "extract_inner_tar.py",
    "explore_data.py",
    "explore_yelp.py",
    "process_yelp.py",
    "show_project_files.py",
    "load_imdb.py",
]

for script in scripts_to_delete:
    delete_file(script)

print("\n" + "="*60)
print("CLEANUP COMPLETE!")
print("="*60)

# Show space saved
print("\nEstimated space saved: ~15 GB")
print("\nFiles kept:")
print("  ✓ Final trained model (final_model/)")
print("  ✓ Cleaned datasets")
print("  ✓ Core training/testing scripts")