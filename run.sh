
#!/bin/sh
# Run bf16 with fp32 matmuls (bf16_fp32 baseline) on folds 0 and 1
gpu=0
results_folder="eval_quantile"
context=100000
n_ensembles=1
inf_batch_size=64
mode=bf16_fp32

# bf16 autocast enabled, matmuls in fp32/TF32 disabled (highest) for bf16_fp32 baseline
export DISABLE_BF16=0
export MATMUL_PRECISION=highest

for fold in 0 1; do
  python paper_evaluation.py \
    --gpu-to-use "${gpu}" \
    --fold "${fold}" \
    --n-ensembles "${n_ensembles}" \
    --context_size "${context}" \
    --precision-tag "${mode}" \
    --inf-batch-size "${inf_batch_size}" \
    --results-folder "${results_folder}"
done
