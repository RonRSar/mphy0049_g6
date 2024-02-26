import os

def get_directory_sizes(directory):
    """
    Calculate the size of each individual subdirectory within the given directory.
    """
    directory_sizes = {}
    for dirpath, dirnames, filenames in os.walk(directory):
        total_size = sum(os.path.getsize(os.path.join(dirpath, filename)) for filename in filenames)
        directory_sizes[dirpath] = total_size
    return directory_sizes

def main():
    # Path to the site-packages directory in your Python environment
    site_packages_path = input("Enter the path to the site-packages directory: ")

    # Check if the provided path exists
    if not os.path.exists(site_packages_path):
        print("Error: The provided path does not exist.")
        return

    # Calculate the size of each individual subdirectory within the site-packages directory
    directory_sizes = get_directory_sizes(site_packages_path)

    # Print the size of each individual subdirectory
    for dirpath, size in sorted(directory_sizes.items()):
        print(f"Size of {dirpath}: {size // (1024 * 1024)} MB")

if __name__ == "__main__":
    main()