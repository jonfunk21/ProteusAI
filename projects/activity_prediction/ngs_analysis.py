from collections import defaultdict
import gzip
import os
import subprocess
import pysam
import time
import json
from tqdm import tqdm

fast_q_path = '/home/jonfunk21/mldecontainer/reads/nanopore/SDT_JL_26_5_23/1/20230526_1400_X2_FAP56552_1a562e98/fastq_pass/'
ngs_dir = '/home/jonfunk21/mldecontainer/ngs'
extracted_dir = os.path.join(ngs_dir, 'extracted')
umi_dest = os.path.join(ngs_dir, 'umi_fasq')

# create directories in blob
if not os.path.exists(ngs_dir):
	os.mkdir(ngs_dir)

if not os.path.exists(extracted_dir):
	os.mkdir(extracted_dir)

if not os.path.exists(umi_dest):
	os.mkdir(umi_dest)

fast_q_files = [f for f in os.listdir(fast_q_path) if f.endswith('.fastq.gz')]
fast_q_files.sort()
reference_fasta = os.path.join(ngs_dir, '../ASMT.fasta')

# index reference file
cmd = f'samtools faidx {reference_fasta}'
result = subprocess.run(cmd.split(), capture_output=True, text=True)

def combine_dicts(dicts):
    combined_dict = defaultdict(list)

    for dictionary in dicts:
        for key, value in dictionary.items():
            combined_dict[key].extend(value)
    
    return combined_dict


def extract_umis(infile, ngs_dir=ngs_dir):
	infile_name = infile.split('/')[-1]
	infile_name = infile_name.split('.')[0]
	extracted_dir = os.path.join(ngs_dir, 'extracted')
	umi_fastq_file = os.path.join(extracted_dir, f'{infile_name}_extracted.fastq.gz')

	# Search for unique molecular identifiers using umi_tools
	cmd = f'umi_tools extract --bc-pattern=NNNNNNNNNNNNNNNNNNNNNNNNN --stdin {infile} --stdout {umi_fastq_file}'
	
	# SPLIT CALCULATION OF FILES AND COLLECTING OF BARCODES
	result = subprocess.run(cmd.split(), capture_output=True, text=True)

# collect all the reads for each umi
def get_mappings(infile, ngs_dir=ngs_dir):
	infile_name = infile.split('/')[-1]
	infile_name = infile_name.split('.')[0]
	extracted_dir = os.path.join(ngs_dir, 'extracted')
	umi_fastq_file = os.path.join(extracted_dir, f'{infile_name}_extracted.fastq.gz')
	
	mapping = {}
	new_read = False
	count = 0
	lines = []
	if os.path.exists(umi_fastq_file):
		write_mode = 'at'
	else:
		write_mode = 'rt'

	with gzip.open(umi_fastq_file, write_mode) as f:
		for i, line in enumerate(f):
			if i == 0 or i % 4 == 0:
				new_read = True
				n = line.split(' ')[0]
				umi = n.split('_')[1]
				#lines.append(line)

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
	return mapping

def get_variants(mapping, ngs_dir=ngs_dir, reference_fasta=reference_fasta, umi_dest=umi_dest):
	# variant calling
	variants = {}
	for key, value in mapping.items():
		# write fastq.gz file for a umi
		umi_fastq = os.path.join(ngs_dir, key + '.fastq.gz')
		with gzip.open(umi_fastq, 'wt') as f:
			for lines in value:
				for line in lines:
					f.write(line)
		
		# use minimap2 for mapping umi.fastq.gz against the reference fasta
		cmd = ['minimap2', '-ax', 'map-ont', reference_fasta, umi_fastq]
		sam_file = os.path.join(ngs_dir, "output.sam")
		with open(sam_file, 'w') as f:
			result = subprocess.run(cmd, stdout=f, text=True)

		# create SAM file
		cmd = ['samtools', 'view', '-S', '-b', os.path.join(ngs_dir, "output.sam")]
		bamfile = os.path.join(ngs_dir, "output.bam")
		with open(bamfile, 'w') as f:
			result = subprocess.run(cmd, stdout=f, text=True)

		# Sort BAM file
		sorted_bamfile = os.path.join(ngs_dir, 'sorted_output.bam')
		cmd = ['samtools', 'sort', bamfile, '-o', sorted_bamfile]
		result = subprocess.run(cmd, capture_output=True, text=True)

		# index BAM file
		cmd = ['samtools', 'index', sorted_bamfile]
		sorted_bamfile_indexed = os.path.join(ngs_dir, "sorted_output.bam.bai")
		with open(sorted_bamfile_indexed, 'w') as f:
			result = subprocess.run(cmd, stdout=f, text=True)

		# Use Longshot
		output_vcf = os.path.join(ngs_dir, 'output.vcf')
		cmd = ['longshot', '--bam', sorted_bamfile, '--ref', reference_fasta, '--out', output_vcf, '-F']
		result = subprocess.run(cmd, capture_output=True, text=True)

		vcf = pysam.VariantFile(output_vcf)
		variants[key] = []
		for variant in vcf:
			variants[key].append(str(variant.ref) + str(variant.pos) + str(variant.alts[0]))
		
		# remove temporary file
		cmd = f'rm {sam_file} {bamfile} {sorted_bamfile} {sorted_bamfile_indexed}'
		result = subprocess.run(cmd.split(), capture_output=True, text=True)

		cmd = f'mv {umi_fastq} {umi_dest}'

	
	return variants 

# test
if __name__ == '__main__':
	umi_fastq_file = '/home/jonfunk21/mldecontainer/ngs/extracted/FAP56552_pass_1a562e98_ebd58ca4_0_extracted.fastq.gz'

	mappings = []
	for i in tqdm(range(len(fast_q_files))):
		infile = os.path.join(fast_q_path, fast_q_files[i])
		extract_umis(infile)
		mapping = get_mappings(infile)
		#mappings.append(mapping)


	# combine mapping of all read files
	#combined_mappings = combine_dicts(mappings)

	# filter out umis which are below the minimum coverage
	#min_coverage = 10
	#filtered_mapping = {}
	#for key, value in combined_mappings.items():
	#	if len(value) // 4 >= min_coverage:
	#		filtered_mapping[key] = combined_mappings[key]

	#variants = get_variants(filtered_mapping)
	#with open('variants.json', 'w') as f:
	#	json.dump(variants, f)