python train.py \
  --checkpoint Helsinki-NLP/opus-mt-es-fi \
  --out_model_name es_fi_quz \
  --extra_data_codes quy quz \
  --epochs 20 \
  --push_to_hub \
  --encoder 0 0 0 0 0 0 \
  --decoder 0.1 0.2 0.3 0.4 0.5 0.6
