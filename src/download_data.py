from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import ensure_dataset, make_dirs

if __name__ == "__main__":
    make_dirs()
    df, status = ensure_dataset(prefer_public=True)
    print("Dataset source:", status["dataset_source"])
    for msg in status["messages"]:
        print("-", msg)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print(df.head(3).to_string(index=False))
