from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def main(results_path:str, output_dir:str):
   # Create output directory
   output_dir = Path(output_dir)
   if not output_dir.exists():
      output_dir.mkdir(parents=True)
   results = pd.read_csv(results_path, index_col=None)
   # BLUE
   fig, ax = plt.subplots(1,1,figsize=(6,6))
   ax = sns.lineplot(results, x='epoch', y='eval_blue', hue='k', palette='tab10', ax=ax)
   ax.set_title('Validation BLEU score by value of K')
   ax.set_ylabel('BLEU (%)')
   ax.set_xlabel('Epoch')
   ax.legend().set_title('K (%)')
   ax.figure.savefig(output_dir/'BLEU.jpeg')

   # Chrf
   fig, ax = plt.subplots(1,1,figsize=(6,6))
   ax = sns.lineplot(results, x='epoch', y='eval_chrf', hue='k', palette='tab10', ax=ax)
   ax.set_title('Validation chrf++ score by value of K')
   ax.set_ylabel('chrf++')
   ax.set_xlabel('Epoch')
   ax.legend().set_title('K (%)')
   ax.figure.savefig(output_dir/'chrf.jpeg')

   # BLUE
   fig, ax = plt.subplots(1,1,figsize=(6,6))
   ax = sns.lineplot(results, x='epoch', y='eval_loss', hue='k', palette='tab10', ax=ax)
   ax.set_title('Validation loss by value of K')
   ax.set_ylabel('Loss')
   ax.set_xlabel('Epoch')
   ax.legend().set_title('K (%)')
   ax.figure.savefig(output_dir/'loss.jpeg')