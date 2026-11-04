"""Turns a KYTY_SAMPLE_OUT file into a report of where the sampled thread spent its time.

usage: py -3 -I tools/sample_report.py <samples.txt> <kyty_emulator.exe> [top]
"""
import bisect
import collections
import subprocess
import sys

SYMBOLIZER = r"C:\Program Files\LLVM\bin\llvm-symbolizer.exe"
PREFERRED_BASE = 0x140000000


def main():
    path, exe = sys.argv[1], sys.argv[2]
    top = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    modules, samples = [], []
    for line in open(path):
        parts = line.split(None, 3)
        if parts and parts[0] == "module":
            modules.append((int(parts[1], 16), int(parts[2], 16), parts[3].strip()))
        elif parts and parts[0] == "sample":
            samples.append([int(x, 16) for x in line.split()[1:]])
    modules.sort()
    bases = [m[0] for m in modules]

    def module_of(address):
        i = bisect.bisect_right(bases, address) - 1
        if i >= 0 and address < modules[i][0] + modules[i][1]:
            return modules[i]
        return None

    main_module = next(m for m in modules if m[2].lower().endswith(exe.split("\\")[-1].lower()))
    wanted = sorted({a - main_module[0] + PREFERRED_BASE for s in samples for a in s
                     if module_of(a) == main_module})
    names = {}
    result = subprocess.run([SYMBOLIZER, "--obj=" + exe, "--no-inlines", "--functions=linkage",
                             "--demangle"], input="\n".join(hex(a) for a in wanted),
                            capture_output=True, text=True, check=True)
    blocks = result.stdout.strip().split("\n\n")
    for address, block in zip(wanted, blocks):
        names[address - PREFERRED_BASE + main_module[0]] = block.split("\n")[0][:90]

    def name(address):
        if address in names:
            return names[address]
        module = module_of(address)
        return module[2].split("\\")[-1] if module else "<jit>"

    leaf = collections.Counter()
    inclusive = collections.Counter()
    for sample in samples:
        if not sample:
            continue
        leaf[name(sample[0])] += 1
        for label in {name(a) for a in sample}:
            inclusive[label] += 1
    total = len(samples)
    print(f"{total} samples")
    print("--- self")
    for label, count in leaf.most_common(top):
        print(f"{100.0 * count / total:5.1f}%  {label}")
    print("--- inclusive")
    for label, count in inclusive.most_common(top):
        print(f"{100.0 * count / total:5.1f}%  {label}")


main()
