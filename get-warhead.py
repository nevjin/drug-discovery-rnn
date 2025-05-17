
input_file = 'sampled_molecules-v4.out'  # replace with your actual file name
output_file = 'sample-with-warheads-modified.out'  # name for the new file containing the filtered lines

# Open the input file and output file
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        if line.startswith("C=CC(=O)N") or line.endswith("C=CC(=O)N"):
            outfile.write(line)
