# util/hdf_reader.py
#
# A module to provide a robust function for reading entire HDF5 files
# into a Python dictionary, preserving the hierarchical structure.
# This function is intended for general HDF5 file inspection.
#

import h5py
import numpy as np
import warnings

# Suppress H5py warnings that might occur during file opening/access
warnings.filterwarnings("ignore", category=UserWarning, module='h5py')

def read_hdf_to_dict_recursive(hdf_group_or_file, current_dict=None):
    """
    Recursively reads an HDF5 group or file into a Python dictionary.

    Args:
        hdf_group_or_file: An h5py.File or h5py.Group object.
        current_dict (dict, optional): The dictionary to populate. Defaults to None.

    Returns:
        dict: A dictionary representation of the HDF5 structure.
    """
    if current_dict is None:
        current_dict = {}

    for key, item in hdf_group_or_file.items():
        if isinstance(item, h5py.Dataset):
            # If it's a dataset, load its content
            value = item[()]
            # Attempt to decode bytes to string if applicable
            if isinstance(value, bytes):
                try:
                    value = value.decode('utf-8')
                except (UnicodeDecodeError, AttributeError):
                    pass # Keep as bytes if decoding fails or not applicable
            current_dict[key] = value
        elif isinstance(item, h5py.Group):
            # If it's a group, create a nested dictionary and recurse
            current_dict[key] = {}
            read_hdf_to_dict_recursive(item, current_dict[key])
    return current_dict

def load_hdf_file_as_dict(filepath):
    """
    Opens an HDF5 file and loads its entire content into a Python dictionary.

    Args:
        filepath (str): The path to the HDF5 file.

    Returns:
        dict: A dictionary representing the HDF5 file's structure and data.
              Returns an empty dictionary if the file cannot be opened or is empty.
    """
    data_dict = {}
    try:
        with h5py.File(filepath, 'r') as f:
            data_dict = read_hdf_to_dict_recursive(f)
    except Exception as e:
        print(f"Error loading HDF5 file '{filepath}': {e}")
    return data_dict

if __name__ == "__main__":
    # This is a simple test case for hdf_reader.py
    # You would typically run 00_hdf_data_inspector_V2.py to see it in action.

    # Create a dummy HDF5 file for testing
    dummy_filepath = "test_dummy_data.hdf"
    with h5py.File(dummy_filepath, 'w') as f:
        f.create_group("metadata")
        f["metadata"].create_dataset("run_id", data=123)
        f["metadata"].create_dataset("comment", data="This is a test run".encode('utf-8'))
        f.create_group("data")
        f["data"].create_dataset("ADC_ch1", data=np.random.rand(5, 10))
        f["data"].create_dataset("FPGAtime", data=np.arange(5))

    print(f"Created dummy file: {dummy_filepath}")

    # Load the dummy file using the new function
    loaded_data = load_hdf_file_as_dict(dummy_filepath)

    print("\n--- Structure of the loaded dummy file ---")
    def print_dict_structure(d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                print(f"{'  ' * indent}- {key}/")
                print_dict_structure(value, indent + 1)
            else:
                print(f"{'  ' * indent}- {key}: {type(value).__name__}, Shape: {getattr(value, 'shape', 'N/A')}")

    print_dict_structure(loaded_data)

    # Clean up dummy file
    import os
    os.remove(dummy_filepath)
    print(f"\nCleaned up dummy file: {dummy_filepath}")