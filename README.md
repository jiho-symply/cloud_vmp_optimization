# cloud_vmp_optimization

Workspace layout:

- `data/metadata`: dataset docs, schemas, bucket definitions, and download links
- `data/raw`: downloaded Azure raw trace files
- `data/processed`: experiment-specific processed samples such as `sample_vm_data.csv`
- `experiments/current`: current working experiment files from the former repo root
- `experiments/version1`: first experiment bundle
- `experiments/version2`: second experiment bundle
- `results/<experiment>/runs`: solver outputs and generated figures per experiment
- `notebooks/reference`: upstream/reference notebooks that analyze the Azure dataset
- `scripts`: helper utilities such as dataset download scripts

Python scripts inside each experiment now resolve their dataset and results paths from the repository root, so they can be run from any working directory.
