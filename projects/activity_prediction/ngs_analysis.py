from collections import defaultdict
import gzip
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import os
import subprocess
import pysam


fast_q_path = '/home/jonfunk21/mldecontainer/reads/nanopore/SDT_JL_26_5_23/1/20230526_1400_X2_FAP56552_1a562e98/fastq_pass/'
fast_q_files = [f for f in os.listdir(fast_q_path) if f.endswith('.fastq.gz')]

ngs_dir = '/home/jonfunk21/ngs'
if not os.path.exists(ngs_dir):
	os.mkdir(ngs_dir)

# test
test_name = fast_q_files[1].split('.')[0]
test_file = os.path.join(fast_q_path, fast_q_files[1])
test_out = os.path.join(ngs_dir, 'extracted.fastq.gz')

# Search for unique molecular identifiers using umi_tools
cmd = f'umi_tools extract --bc-pattern=NNNNNNNNNNNNNNNNNNNNNNNNN --stdin {test_file} --stdout {test_out}'
#result = subprocess.run(cmd.split(), capture_output=True, text=True)

# collect all the reads for each umi
fastq_file = test_out
mapping = {}
new_read = False
count = 0
lines = []
with gzip.open(fastq_file, 'rt') as f:
	for i, line in enumerate(f):
		if i == 0 or i % 4 == 0:
			new_read = True
			n = line.split(' ')[0]
			umi = n.split('_')[1]
			lines.append(line)

		elif new_read:
			lines.append(line)
			count += 1
		
		if count == 3:
			count = 0
			if umi in mapping.keys():
				mapping[umi].append(lines)
			else:
				mapping[umi] = lines
			lines = []

# filter out umis which are below the minimum coverage
min_coverage = 10
filtered_mapping = {}
for key, value in mapping.items():
	if len(value) // 4 >= min_coverage:
		filtered_mapping[key] = mapping[key]

# index reference file
reference_fasta = os.path.join(ngs_dir, 'ASMT.fasta')
cmd = f'samtools faidx {reference_fasta}'
result = subprocess.run(cmd.split(), capture_output=True, text=True)

# variant calling
variants = {}
i == 0
for key, value in filtered_mapping.items():
	# write fastq.gz file
	f_name = os.path.join(ngs_dir, key)
	with gzip.open(f'{f_name}.fastq.gz', 'wt') as f:
		for lines in value:
			for line in lines:
				f.write(line)
	
	# use minimap2 for mapping against the reference
	cmd = ['minimap2', '-ax', 'map-ont', reference_fasta, f'{f_name}.fastq.gz']
	with open(os.path.join(ngs_dir, "output.sam"), 'w') as f:
		result = subprocess.run(cmd, stdout=f, text=True)
	
	# create BAM file
	cmd = ['samtools', 'view', '-S', '-b', os.path.join(ngs_dir, "output.sam")]
	with open(os.path.join(ngs_dir, "output.bam"), 'w') as f:
		result = subprocess.run(cmd, stdout=f, text=True)

	# Sort BAM file
	cmd = ['samtools', 'sort', os.path.join(ngs_dir, 'output.bam'), '-o', os.path.join(ngs_dir, 'sorted_output.bam')]
	result = subprocess.run(cmd, capture_output=True, text=True)

	# index BAM file
	cmd = ['samtools', 'index', os.path.join(ngs_dir, "sorted_output.bam")]
	with open(os.path.join(ngs_dir, "sorted_output.bam.bai"), 'w') as f:
		result = subprocess.run(cmd, stdout=f, text=True)

	# Use Longshot
	cmd = ['longshot', '--bam', os.path.join(ngs_dir, 'sorted_output.bam'), '--ref', reference_fasta, '--out', os.path.join(ngs_dir, 'output.vcf'), '-F']
	result = subprocess.run(cmd, capture_output=True, text=True)

	vcf = pysam.VariantFile(os.path.join(ngs_dir, 'output.vcf'))
	variants[key] = []
	for variant in vcf:
		variants[key].append(str(variant.ref) + str(variant.pos) + str(variant.alts[0]))
	
	# remove temporary file
	cmd = f'rm {os.path.join(ngs_dir, f_name + ".fastq.gz")} {os.path.join(ngs_dir, "output.sam")} {os.path.join(ngs_dir, "output.bam")} {os.path.join(ngs_dir, "sorted_output.bam")} {os.path.join(ngs_dir, "output.vcf")}'
	result = subprocess.run(cmd.split(), capture_output=True, text=True)

print(set(filtered_mapping.keys()))
print(variants)