import os, shutil, sys

if __name__ == '__main__':
    if sys.argv[1] == 'rmtree':
        for d in sys.argv[2:]:
            if os.path.isdir(d):
                shutil.rmtree(d); print('deleted', d)
