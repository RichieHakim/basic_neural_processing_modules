"""
Helper functions for the osfclient package: https://github.com/osfclient/osfclient
"""

def build_folder_tree(folder, include_files=True, verbose=False):
    """
    Builds a hierarchical dictionary of the folder tree of a project.
    Recursively calls itself to build the tree.
    RH 2022
    
    Args:
        folder (osfclient.models.file.Folder):
            Folder object to build tree from.
            Can be made by:
                client = osfclient.OSF()
                proj = client.project('kuh6q or whatever your project ID is')
                folder = proj.storage()
        include_files (bool):
            Whether to include files in the tree as the leaves of the tree.
        verbose (bool):
            Whether to print the the current folder or file being processed.

    Returns:
        tree (dict):
            Dictionary of the folder tree.
            If include_files is True, then the leaves of the tree are the files.
            Otherwise, the leaves are the folders.
    """
    tree = {}
    for sub in folder.folders:
        tree[sub.name] = build_folder_tree(sub, verbose=verbose)
        print(sub.name) if verbose else None
    if include_files:
        for file in folder.files:
            tree[file.name] = file
            print(file.name) if verbose else None
    return tree


def get_folder(folder, path):
    """
    Returns the folder object at the given path.
    RH 2022

    Args:
        folder (osfclient.models.file.Folder):
            Folder object to search.
            Can be made by:
                client = osfclient.OSF()
                proj = client.project('kuh6q or whatever your project ID is')
                folder = proj.storage()
        path (list of strings):
            Path to the folder to return.
            Each string in the list is a folder name.
            The last string is the folder name to return.

    Returns:
        folder (osfclient.models.file.Folder):
            Folder object at the given path.
    """
    if len(path) == 0:
        return folder
    
    for sub in folder.folders:
        if sub.name == path[0]:
            return get_folder(sub, path[1:])