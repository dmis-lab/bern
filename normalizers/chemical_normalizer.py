import os
import socket
import string


CHEMICAL_PORT = 18890
chem2oid = None


def run_server(logic_func, port):
    host = '127.0.0.1'
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen(1)
        while True:
            conn, addr = s.accept()
            data = conn.recv(1024).decode('utf-8')
            if data == 'quit':
                break
            logic_func(data, addr)
            conn.send(b'Done')
            conn.close()


def run_normalizer(data, _):
    args = data.split('\t')
    assert len(args) == 5
    input_dir, input_filename, output_dir, output_filename, dict_path = args

    global chem2oid
    if chem2oid is None:
        # Create dictionary for exact match
        chem2oid = dict()
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                oid, names = line[:-1].split('||')
                names = names.split('|')
                for name in names:
                    # a part of tmChem normalization
                    chem2oid[get_tmchem_name(name)] = oid

    oids = list()
    # Read names and get oids
    with open(os.path.join(input_dir, input_filename), 'r',
              encoding='utf-8') as f:
        for line in f:
            name = line[:-1]

            # a part of tmChem normalization
            normalized_name = get_tmchem_name(name)

            if normalized_name in chem2oid:
                oids.append(chem2oid[normalized_name])
            else:
                oids.append('CUI-less')

    # Write to file
    with open(os.path.join(output_dir, output_filename), 'w',
              encoding='utf-8') as f_out:
        for oid in oids:
            f_out.write(oid + '\n')


def get_tmchem_name(name):
    # 1. lowercase, 2. removes all whitespace and punctuation
    # https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-7-S1-S3
    normalized_name = ''
    for nc in name.lower():
        if nc == ' ' or nc in string.punctuation:
            continue
        normalized_name += nc
    return normalized_name


if __name__ == "__main__":
    run_server(run_normalizer, CHEMICAL_PORT)
