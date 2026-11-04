"""Turns a KYTY_FN_PROFILE file into a table of where the GPU thread's time went.

usage: py -3 -I tools/function_report.py <profile.txt> <kyty_emulator.exe> <frame ms> [top]
"""
import subprocess
import sys

SYMBOLIZER = r"C:\Program Files\LLVM\bin\llvm-symbolizer.exe"


def main():
    path, exe, frame_ms = sys.argv[1], sys.argv[2], float(sys.argv[3])
    top = int(sys.argv[4]) if len(sys.argv) > 4 else 50
    window = 1
    rows = []
    for line in open(path):
        parts = line.split()
        if parts[0] == "ticks":
            window = int(parts[1])
        else:
            rows.append((int(parts[1], 16), int(parts[2]), int(parts[3]), int(parts[4])))
    result = subprocess.run([SYMBOLIZER, "--obj=" + exe, "--no-inlines", "--functions=linkage", "--demangle"],
                            input="\n".join(hex(r[0]) for r in rows), capture_output=True, text=True, check=True)
    names = [b.split("\n")[0] for b in result.stdout.strip().split("\n\n")]
    def clean(name):
        for text in ("Libs::Graphics::", "class ", "struct ", "ShaderRecompiler::"):
            name = name.replace(text, "")
        return name[:66]
    ms = lambda ticks: ticks / window * frame_ms
    named = [(ms(r[2]), ms(r[3]), r[1], clean(n)) for r, n in zip(rows, names)]
    print(f"sum of self: {sum(x[0] for x in named):.1f} ms of a {frame_ms:.1f} ms frame")
    print("self ms | total ms | calls in window | function (by self)")
    for row in sorted(named, reverse=True)[:top]:
        print(f"{row[0]:7.2f} {row[1]:8.2f} {row[2]:9d}  {row[3]}")
    print("--- by total")
    for row in sorted(named, key=lambda x: -x[1])[:top]:
        print(f"{row[0]:7.2f} {row[1]:8.2f} {row[2]:9d}  {row[3]}")


main()
