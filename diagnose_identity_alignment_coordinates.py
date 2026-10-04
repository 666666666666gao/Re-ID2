"""Unchanged frozen cross-coordinate protocol with accurate M2b metadata."""
import diagnose_cross_identity_coordinates as diagnostic

original_write = diagnostic.write_json


def write_result(path,value):
    if path.name == 'result.json':
        value['protocol'] = value['protocol'].replace('Fixed original M2 best.', 'Fixed original M2b best.')
    original_write(path,value)


diagnostic.write_json = write_result

if __name__ == '__main__':
    diagnostic.main()
