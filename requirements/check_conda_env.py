import os


def main():
    # Check if conda env is active
    conda_default_env = os.environ.get("CONDA_DEFAULT_ENV")

    if not conda_default_env or conda_default_env == "base":
        print("Error: Activate a Conda environment other than 'base'")
        exit(1)


if __name__ == "__main__":
    main()
