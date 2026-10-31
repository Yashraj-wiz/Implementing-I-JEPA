#!/usr/bin/env bash
# Runs the oracle-decoder ablation sequentially, then the fixed-benchmark evaluation.
# Safe to rerun: each training resumes from its latest.pt.
set -euo pipefail
cd "$(dirname "$0")/.."
for mode in predicted oracle; do
  python -m src.ijepa.train_decoder --config "configs/stl10_decoder_ablation_${mode}_enc200k_50k.yaml" \
    > "artifacts/stl10_decoder_ablation_${mode}.log" 2>&1
done
python -m src.ijepa.evaluate_decoder \
  --encoder artifacts/modal_encoder_step_0200000_final.pt \
  --decoder predicted=artifacts/stl10_decoder_ablation/predicted_enc200k_50k/step_0050000.pt \
  --decoder oracle=artifacts/stl10_decoder_ablation/oracle_enc200k_50k/step_0050000.pt \
  --decoder adapted_reference=artifacts/modal_decoder_final_encoder_adapt_step_0050000_final.pt \
  --output-dir artifacts/stl10_decoder_ablation/evaluation --num-workers 2 --with-shuffled \
  > artifacts/stl10_decoder_ablation_eval.log 2>&1
echo DONE
