from src import data_prep, labels, predict, train


def main() -> None:
    steps = [
        ("1/4 Data preparation", data_prep.main),
        ("2/4 Labels", labels.main),
        ("3/4 Training", train.main),
        ("4/4 Scoring open applications", predict.main),
    ]
    for name, step in steps:
        print(f"\n=== {name} ===")
        step()
    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
