# Snakefile
import os

workdir: '/path/to/GEN_TEST'

# variable
base_path = '/path/to'
cohort = 'GEN_TEST'
date = 'YYMMDD'
h5ad_path = '/path/to/GEN_TEST/Single_cell_files'
manifest = 'GEN_TEST_manifest.txt'
batch = 'libraryID'

# additional variable
cluster_name = 'subtype'
exclude_file = '/path/to/GEN_TEST/YYMMDD_GEN_TEST_pass2_exclude.txt'

rule all:
    input:
        os.path.join(base_path, cohort, date+'_'+cohort, cohort+'_pass3.h5ad')

rule subset:
    input:
        os.path.join(base_path, cohort, date+'_'+cohort, cohort+'_pass2.h5ad')
    output:
        os.path.join(base_path, cohort, date+'_'+cohort, cohort+'_pass2_subset.h5ad')
    conda:
        'py312'
    shell:
        'python 08_subset.py --input {input} --output {output} --cluster {cluster_name} --exclude {exclude_file} >> {cohort}_pass3.log 2>&1'

rule pass3:
    input:
        os.path.join(base_path, cohort, date+'_'+cohort, cohort+'_pass2_subset.h5ad')
    output:
        os.path.join(base_path, cohort, date+'_'+cohort, cohort+'_pass3.h5ad')
    params:
        prefix=os.path.join(base_path, cohort, date+'_'+cohort, cohort+'_pass3')
    conda:
        'py312'
    shell:
        'python 06_iclust.py --input {input} --output {output} --prefix {params.prefix} --batch {batch} >> {cohort}_pass3.log 2>&1'
