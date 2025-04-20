from utils.graph_results import main as graph_results
from utils.parse_results import main as parse_results

def main(log_files_path, results_output_path, plots_output_path):
   parse_results(log_files_path=log_files_path, output_path=results_output_path)
   graph_results(results_path=results_output_path, output_dir=plots_output_path)

if __name__ == '__main__':
   main(
      '.', 'results.csv', 'visualizations/'
   )