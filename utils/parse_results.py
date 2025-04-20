import re
from pathlib import Path
import numpy as np
import pandas as pd

def main(log_files_path: str, output_path:str = 'results.csv'):
   # Get k values
   file_pattern = re.compile(r'lt-sft-K=(\d+).txt')
   log_files = [
      file
      for file in Path(log_files_path).iterdir()
      if file.name.startswith('lt-sft-K')
   ]
   k_values = [
      re.match(file_pattern, file.name).group(1)
      for file in log_files
   ]
   k_values = [int(val) for val in k_values]
   print("Results found: ", k_values)
   # Parse contents
   line_pattern = re.compile(r"{'eval_loss': (\d+(?:\.\d+)?), 'eval_bleu': (\d+(?:\.\d+)?), 'eval_chrf': (\d+(?:\.\d+)?), 'eval_gen_len': .+, 'eval_runtime': .+, 'eval_samples_per_second': .+, 'eval_steps_per_second': .+, 'epoch': (\d+(?:\.\d+)?)}")
   val_results = []
   for file, k in zip(log_files, k_values):
      with file.open('r') as log_file:
         match_lines = [re.search(line_pattern, line) for line in log_file.readlines()]
         val_results.extend([(k,) + match.groups() for match in match_lines if match])
   # Save contents as csv
   val_results = pd.DataFrame(
      val_results, columns=['k', 'eval_loss', 'eval_blue', 'eval_chrf', 'epoch']
   )
   val_results.to_csv(output_path, index=False)