import pytest
from utils.parse_results import main

def test_parse_results(log_files_path):
   main(log_files_path=log_files_path)

@pytest.fixture
def log_files_path(tmp_path):
   with (tmp_path / 'lt-sft-K=10.txt').open('w') as file:
      file.write("""Data read from original/jw300_quy | Count: 125008 | 125008
Data read from original/dict_misc_quy | Count: 9000 | 9000
Data read from original/minedu_quy | Count: 643 | 643
Data read from extra/bible_quy | Count: 34831 | 34831
Data read from extra/bol_const_quy | Count: 2194 | 2194
Data read from extra/per_const_quy | Count: 1277 | 1277
Data read from extra/lexicon_quy | Count: 6161 | 6161
Data read from extra/handbook_quy | Count: 2297 | 2298
Data read from extra/web_misc_quy | Count: 985 | 985
Data read from extra/tatoeba_quy | Count: 163 | 163
Data read from extra/reglamento_quz | Count: 287 | 287
Data read from extra/cosude_quz | Count: 529 | 529
Data read from extra/dw_quz | Count: 856 | 856
Data read from extra/fundacion_quz | Count: 440 | 440
Lang: Quechua | Train Count: 184671 | Dev Count: 996
{'loss': 9.0557, 'learning_rate': 1.9955346650998824e-05, 'epoch': 0.05}
{'loss': 8.4287, 'learning_rate': 1.9910150953629216e-05, 'epoch': 0.09}
{'eval_loss': 8.925552368164062, 'eval_bleu': 0.939, 'eval_chrf': 11.8729, 'eval_gen_len': 12.9497, 'eval_runtime': 24.1276, 'eval_samples_per_second': 41.198, 'eval_steps_per_second': 2.611, 'epoch': 0.09}
{'loss': 7.8879, 'learning_rate': 1.9864955256259607e-05, 'epoch': 0.14}
{'loss': 7.3476, 'learning_rate': 1.9819759558889995e-05, 'epoch': 0.18}
{'eval_loss': 7.919479846954346, 'eval_bleu': 0.9664, 'eval_chrf': 11.781, 'eval_gen_len': 13.0443, 'eval_runtime': 24.35, 'eval_samples_per_second': 40.821, 'eval_steps_per_second': 2.587, 'epoch': 0.18}
{'loss': 6.8117, 'learning_rate': 1.9774654252915124e-05, 'epoch': 0.23}"""
      )
   with (tmp_path / 'lt-sft-K=20.txt').open('w') as file:
      file.write("""Data read from original/jw300_quy | Count: 125008 | 125008
Data read from original/dict_misc_quy | Count: 9000 | 9000
Data read from original/minedu_quy | Count: 643 | 643
Data read from extra/bible_quy | Count: 34831 | 34831
Data read from extra/bol_const_quy | Count: 2194 | 2194
Data read from extra/per_const_quy | Count: 1277 | 1277
Data read from extra/lexicon_quy | Count: 6161 | 6161
Data read from extra/handbook_quy | Count: 2297 | 2298
Data read from extra/web_misc_quy | Count: 985 | 985
Data read from extra/tatoeba_quy | Count: 163 | 163
Data read from extra/reglamento_quz | Count: 287 | 287
Data read from extra/cosude_quz | Count: 529 | 529
Data read from extra/dw_quz | Count: 856 | 856
Data read from extra/fundacion_quz | Count: 440 | 440
Lang: Quechua | Train Count: 184671 | Dev Count: 996
{'loss': 9.0557, 'learning_rate': 1.9955346650998824e-05, 'epoch': 0.05}
{'loss': 8.4287, 'learning_rate': 1.9910150953629216e-05, 'epoch': 0.09}
{'eval_loss': 8.1, 'eval_bleu': 9, 'eval_chrf': 11.8729, 'eval_gen_len': 12.9497, 'eval_runtime': 24.1276, 'eval_samples_per_second': 41.198, 'eval_steps_per_second': 2.611, 'epoch': 0.09}
{'loss': 7.8879, 'learning_rate': 1.9864955256259607e-05, 'epoch': 0.14}
{'loss': 7.3476, 'learning_rate': 1.9819759558889995e-05, 'epoch': 0.18}
{'eval_loss': 7.919479846954346, 'eval_bleu': 0.9664, 'eval_chrf': 11.781, 'eval_gen_len': 13.0443, 'eval_runtime': 24.35, 'eval_samples_per_second': 40.821, 'eval_steps_per_second': 2.587, 'epoch': 0.18}
{'loss': 6.8117, 'learning_rate': 1.9774654252915124e-05, 'epoch': 0.23}"""
      )
   return tmp_path
