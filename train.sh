cd src/machine_translation
python mt_model.py --checkpoint Helsinki-NLP/opus-mt-es-fi \
   --out_model_name es_fi_quz --extra_data_codes quy quz --epochs 20 \
   --push_to_hub
cd ../..