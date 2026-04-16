"""Run from project root (C:\\DEVELOP\\LCM_AI) to find .py files containing null bytes."""
import os
import sys

def main():
    start = os.path.abspath(os.curdir)
    found = []
    for root, dirs, files in os.walk(start):
        dirs[:] = [d for d in dirs if d not in ('.git', 'node_modules')]
        for f in files:
            if f.endswith('.py'):
                path = os.path.join(root, f)
                try:
                    with open(path, 'rb') as fp:
                        if b'\x00' in fp.read():
                            found.append(path)
                except Exception as e:
                    print(path, e, file=sys.stderr)
    for p in found:
        print(p)
    if not found:
        print("No .py files with null bytes found in", start)
    return 1 if found else 0

if __name__ == '__main__':
    sys.exit(main())
