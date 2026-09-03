import sys
import os

with open("argv_log.txt", "w") as f:
    f.write("ARGV: " + str(sys.argv) + "\n")
