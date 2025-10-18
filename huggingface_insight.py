from datasets import get_dataset_config_names

# Specify the dataset repository ID
dataset_name = "Zihan1004/FNSPID"

# Fetch the list of configuration names
config_names = get_dataset_config_names(dataset_name)

# Print the available names
print(f"Available configurations for {dataset_name}:")
print(config_names)