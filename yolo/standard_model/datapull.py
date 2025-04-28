import kagglehub

# Download latest version
path = kagglehub.dataset_download("aladdinpersson/pascalvoc-yolo")

print("Path to dataset files:", path)