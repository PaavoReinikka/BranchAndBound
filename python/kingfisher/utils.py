"""
Kingfisher Python Utilities
Provides conversion functions between dense binary data and sparse transaction data.
"""

def dense_to_sparse(dense_data, column_names=None):
    """
    Converts a dense 2D list (rows x columns) into sparse transaction format.
    
    Args:
        dense_data: List of lists/iterables containing 0/1 or True/False.
        column_names: Optional list of strings for attribute names.
        
    Returns:
        sparse_data: List of lists containing indices of active attributes.
        id_to_name: Dictionary mapping index -> name.
        name_to_id: Dictionary mapping name -> index.
    """
    sparse_data = []
    num_cols = len(dense_data[0]) if dense_data else 0
    
    # Generate or validate names
    if column_names is None:
        column_names = [f"Attr_{i}" for i in range(num_cols)]
    
    if len(column_names) != num_cols:
        raise ValueError(f"Number of column names ({len(column_names)}) must match data columns ({num_cols})")
    
    id_to_name = {i: name for i, name in enumerate(column_names)}
    name_to_id = {name: i for i, name in id_to_name.items()}
    
    for row in dense_data:
        sparse_row = [i for i, val in enumerate(row) if val]
        sparse_data.append(sparse_row)
        
    return sparse_data, id_to_name, name_to_id

def sparse_to_dense(sparse_data, k):
    """
    Converts sparse transaction data back to a dense binary matrix.
    
    Args:
        sparse_data: List of lists of indices.
        k: Maximum attribute index.
        
    Returns:
        dense_data: List of lists of integers (0 or 1).
    """
    dense_data = []
    num_cols = k + 1
    
    for sparse_row in sparse_data:
        dense_row = [0] * num_cols
        for idx in sparse_row:
            if idx < num_cols:
                dense_row[idx] = 1
        dense_data.append(dense_row)
        
    return dense_data

def load_names_to_mapping(names_file):
    """
    Loads a text file of names into mapping dictionaries.
    """
    with open(names_file, 'r') as f:
        names = [line.strip() for line in f if line.strip()]
    
    id_to_name = {i: name for i, name in enumerate(names)}
    name_to_id = {name: i for i, name in id_to_name.items()}
    return id_to_name, name_to_id
