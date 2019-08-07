import sys
sys.path.insert(0, '../.')
from utils.compare_plots import compare_plots_best_performing, plot_single_model_multiple_epoch, plot_multiple_model_multiple_epoch

paths_to_compare = ["~/Github/Fully_factorizable_CVAE/outputs/baseline_results/2019:08:06:15:55:41/", 
                    "~/Github/Fully_factorizable_CVAE/outputs/baseline_results_2/2019:08:06:15:55:47/",
                    "~/Github/Fully_factorizable_CVAE/outputs/baseline_results_3/2019:08:06:15:56:09/"]

csv = {'save_dir': [], 'csv_path': []}

for j in range(len(paths_to_compare)):
    csv['save_dir'].append(paths_to_compare[j])
    csv['csv_path'].append(csv['save_dir'][j] + "costs.csv")

# Plot KLD vs MSE for multiple models and numbers of conditions
compare_plots_best_performing(csv)

# Plot KLD vs MSE for multiple models, numbers of conditions and epochs
plot_multiple_model_multiple_epoch(csv)

# CHOOSE SINGLE CSV
single_csv = {'save_dir': csv['save_dir'][0], 'csv_path': csv['csv_path'][0]}

# Plot KLD vs MSE for multiple number of conditions and epochs for a single model
plot_single_model_multiple_epoch(single_csv)

