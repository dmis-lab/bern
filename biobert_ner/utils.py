import time

# https://stackoverflow.com/a/3620972
PROF_DATA = {}


class Profile(object):
    def __init__(self, prefix):
        self.prefix = prefix

    def __call__(self, fn):
        def with_profiling(*args, **kwargs):
            global PROF_DATA
            start_time = time.time()
            ret = fn(*args, **kwargs)

            elapsed_time = time.time() - start_time
            key = '[' + self.prefix + '].' + fn.__name__

            if key not in PROF_DATA:
                PROF_DATA[key] = [0, list()]
            PROF_DATA[key][0] += 1
            PROF_DATA[key][1].append(elapsed_time)

            return ret

        return with_profiling


def show_prof_data():
    for fname, data in sorted(PROF_DATA.items()):
        max_time = max(data[1])
        avg_time = sum(data[1]) / len(data[1])
        total_time = sum(data[1])
        print("\n{} -> called {} times".format(fname, data[0]))
        print("Time total: {:.3f}, max: {:.3f}, avg: {:.3f}".format(
            total_time, max_time, avg_time))


def clear_prof_data():
    global PROF_DATA
    PROF_DATA = {}
