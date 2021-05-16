import os

def read_file(of_path):
    with open(of_path, encoding='latin1') as doc:
        doc = doc.read().lower()
        _header, _blankline, body = doc.partition('\n\n')
        return body

def read_files(in_path):
    file_names = os.listdir(in_path)
    result = {}
    for name in file_names:
        path = os.path.join(in_path, name)
        result[name] = read_files(path) if os.path.isdir(path) else read_file(path)
    return result
