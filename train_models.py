import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_models import train_all_models, find_dataset_file, LABELS, CNN_METRICS_PATH, MODELS_DIR

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", default=None)
    parser.add_argument("--cnn-test-csv", default=None)
    args = parser.parse_args()
    csv_path = args.csv
    if not csv_path:
        csv_path = find_dataset_file()
        if csv_path:
            print(f"Dataset found : {csv_path}")
        else:
            print("ERROR: No dataset found in backend/data/")
            sys.exit(1)
    elif not os.path.isfile(csv_path):
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)
    print("\nTraining KNN, SVM, XGBoost + generating CNN metrics...\n")
    out = train_all_models(csv_path=csv_path, cnn_test_csv=args.cnn_test_csv)
    if not out.get("ok"):
        print(f"ERROR: {out.get('error')}")
        sys.exit(1)
    print("\n==================================================")
    print("  TRAINING COMPLETE")
    print("==================================================")
    print(f"  Samples : {out.get('n_samples')}")
    print()
    acc = out.get("accuracies", {})
    for k in ["cnn", "xgboost", "svm", "knn"]:
        v = acc.get(k)
        if v is not None:
            print(f"  {k.upper():8s}  {v*100:6.2f}%")
    print()
    print(f"  Models  -> {MODELS_DIR}")
    print(f"  CNN     -> {CNN_METRICS_PATH}")

if __name__ == "__main__":
    main()
