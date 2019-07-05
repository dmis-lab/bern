import os
import socket


SPECIES_PORT = 18889
species2oid = None
# os.chdir('../../../')


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
    # client_port = addr[-1]
    args = data.split(' ')
    assert len(args) == 3
    input_dir, output_dir, dict_path = args

    global species2oid
    if species2oid is None:
        # Create dictionary for exact match
        species2oid = {}
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                oid, names = line[:-1].split('||')
                names = names.split('|')
                for name in names:
                    species2oid[name] = oid
    
    for filename in os.listdir(input_dir):
        oids = []
        # Read names and get oids
        with open(os.path.join(input_dir, filename), 'r',
                  encoding='utf-8') as f:
            for line in f:
                name = line[:-1]
                if name in species2oid:
                    oids.append(species2oid[name])
                elif name.lower() in species2oid:
                    oids.append(species2oid[name.lower()])
                else:
                    oids.append('CUI-less')

        # Write to file
        with open(os.path.join(
                output_dir, os.path.splitext(filename)[0] + '.oid'), 'w',
                encoding='utf-8') as f:
            for oid in oids:
                f.write(oid + '\n')
    

if __name__ == "__main__":
    run_server(run_normalizer, SPECIES_PORT)
