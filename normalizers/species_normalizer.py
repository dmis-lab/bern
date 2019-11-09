import os
import socket


SPECIES_PORT = 18889
species2oid = None


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

    global species2oid
    if species2oid is None:
        # Create dictionary for exact match
        species2oid = dict()
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                # only 1 || exists in a line?
                oid, names = line[:-1].split('||')
                names = names.split('|')
                for name in names:
                    species2oid[name] = oid

    oids = list()
    # Read names and get oids
    with open(os.path.join(input_dir, input_filename), 'r', encoding='utf-8') \
            as f:
        for line in f:
            name = line[:-1]
            if name in species2oid:
                oids.append(species2oid[name])
            elif name.lower() in species2oid:
                oids.append(species2oid[name.lower()])
            else:
                oids.append('CUI-less')

    # Write to file
    with open(os.path.join(output_dir, output_filename), 'w',
              encoding='utf-8') as f_out:
        for oid in oids:
            f_out.write(oid + '\n')


if __name__ == "__main__":
    run_server(run_normalizer, SPECIES_PORT)
