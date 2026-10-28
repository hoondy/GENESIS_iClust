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
annotation_file = '/path/to/GEN_TEST/YYMMDD_GEN_TEST_pass3_annotation.csv'

rule all:
    input:
        os.path.join(base_path, cohort, date+'_'+cohort, cohort+'_rc2.h5ad')

rule anno:
    input:
        os.path.join(base_path, cohort, date+'_'+cohort, cohort+'_pass3.h5ad')
    output:
        os.path.join(base_path, cohort, date+'_'+cohort, cohort+'_pass3_anno.h5ad')
    params:
        pal=os.path.join(base_path, cohort, 'annotation/260122_GENESIS_color_palette.csv')
    conda:
        'py312'
    shell:
        'python 09_annotate.py --input {input} --output {output} --annotation {annotation_file} --harmony_batch {batch} --hvf_n_top_genes 6000 --palette {params.pal} >> {cohort}_rc2.log 2>&1'

rule clean:
    input:
        os.path.join(base_path, cohort, date+'_'+cohort, cohort+'_pass3_anno.h5ad')
    output:
        os.path.join(base_path, cohort, date+'_'+cohort, cohort+'_pass3_anno_clean.h5ad')
    params:
        pal=os.path.join(base_path, cohort, 'annotation/260122_GENESIS_color_palette.csv')
    conda:
        'py312'
    shell:
        'python 10_clean.py --input {input} --output {output} --palette {params.pal} >> {cohort}_rc2.log 2>&1'

rule umap:
    input:
        os.path.join(base_path, cohort, date+'_'+cohort, cohort+'_pass3_anno_clean.h5ad')
    output:
        os.path.join(base_path, cohort, date+'_'+cohort, cohort+'_rc2.h5ad')
    params:
        pal=os.path.join(base_path, cohort, 'annotation/260122_GENESIS_color_palette.csv')
    conda:
        'py312'
    shell:
        'python 11_umap.py --input {input} --output {output} --harmony_batch {batch} --hvf_n_top_genes 6000 --palette {params.pal} >> {cohort}_rc2.log 2>&1'
