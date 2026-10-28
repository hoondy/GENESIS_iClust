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

rule all:
    input:
        os.path.join(base_path, cohort, date+'_'+cohort, cohort+'_pass2.h5ad')

rule aggregate:
    output:
        os.path.join(base_path, cohort, date+'_'+cohort, cohort+'_merged.h5ad')
    conda:
        'py312'
    shell:
        'python 01_aggregate.py --path {h5ad_path} --manifest {manifest} --output {output} >> {cohort}_pass1.log 2>&1'

rule annotation:
    input:
        os.path.join(base_path, cohort, date+'_'+cohort, cohort+'_merged.h5ad')
    output:
        os.path.join(base_path, cohort, date+'_'+cohort, cohort+'_merged_anno.h5ad')
    params:
        anno=os.path.join(base_path, cohort, 'annotation/all_genes_EnsemblInfo_104.csv')
    conda:
        'py312'
    shell:
        'python 02_anno.py --input {input} --output {output} --anno {params.anno} >> {cohort}_pass1.log 2>&1'

rule qc:
    input:
        os.path.join(base_path, cohort, date+'_'+cohort, cohort+'_merged_anno.h5ad')
    output:
        os.path.join(base_path, cohort, date+'_'+cohort, cohort+'_merged_anno_qc.h5ad')
    params:
        prefix=os.path.join(base_path, cohort, date+'_'+cohort, cohort+'_qc'),
        anno=os.path.join(base_path, cohort, 'annotation/Human.MitoCarta3.0.csv'),
        mad_k=3,
        pct_mito=5,
        min_n_cells=50
    conda:
        'py312'
    shell:
        'python 03_qc.py --input {input} --output {output} --prefix {params.prefix} --anno {params.anno} --mad_k {params.mad_k} --pct_mito {params.pct_mito} --min_n_cells {params.min_n_cells} >> {cohort}_pass1.log 2>&1'

rule normalization:
    input:
        os.path.join(base_path, cohort, date+'_'+cohort, cohort+'_merged_anno_qc.h5ad')
    output:
        os.path.join(base_path, cohort, date+'_'+cohort, cohort+'_merged_anno_qc_norm.h5ad')
    conda:
        'py312'
    shell:
        'python 04_norm.py --input {input} --output {output} >> {cohort}_pass1.log 2>&1'

rule signature_score:
    input:
        os.path.join(base_path, cohort, date+'_'+cohort, cohort+'_merged_anno_qc_norm.h5ad')
    output:
        os.path.join(base_path, cohort, date+'_'+cohort, cohort+'_merged_anno_qc_norm_sig.h5ad')
    conda:
        'py312'
    shell:
        'python 05_score.py --input {input} --output {output} >> {cohort}_pass1.log 2>&1'

rule pass1:
    input:
        os.path.join(base_path, cohort, date+'_'+cohort, cohort+'_merged_anno_qc_norm_sig.h5ad')
    output:
        os.path.join(base_path, cohort, date+'_'+cohort, cohort+'_pass1.h5ad')
    params:
        prefix=os.path.join(base_path, cohort, date+'_'+cohort, cohort+'_pass1')
    conda:
        'py312'
    shell:
        'python 06_iclust.py --input {input} --output {output} --prefix {params.prefix} --batch {batch} >> {cohort}_pass1.log 2>&1'

rule scrublet:
    input:
        os.path.join(base_path, cohort, date+'_'+cohort, cohort+'_pass1.h5ad')
    output:
        os.path.join(base_path, cohort, date+'_'+cohort, cohort+'_pass1_dedup.h5ad')
    conda:
        'py312'
    shell:
        'python 07_scrublet.py --input {input} --output {output} >> {cohort}_pass1.log 2>&1'

rule pass2:
    input:
        os.path.join(base_path, cohort, date+'_'+cohort, cohort+'_pass1_dedup.h5ad')
    output:
        os.path.join(base_path, cohort, date+'_'+cohort, cohort+'_pass2.h5ad')
    params:
        prefix=os.path.join(base_path, cohort, date+'_'+cohort, cohort+'_pass2')
    conda:
        'py312'
    shell:
        'python 06_iclust.py --input {input} --output {output} --prefix {params.prefix} --batch {batch} >> {cohort}_pass2.log 2>&1'