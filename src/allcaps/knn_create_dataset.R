# Stage 1 of the per-serotype novelty threshold analysis.
#
# Collects the leave-one-serotype-out kNN sweep CSVs and writes one `k{K}_knn_data.csv`
# per value of k, joined to a ground-truth table. Stage 2 is knn_calculate_novel.R.
#
# Invoked by `ALLCAPS knn`; runnable directly as:
#   Rscript knn_create_dataset.R --knn-raw <dir> --ground-truth <csv> --output <dir>
#
# This used to locate its inputs through rstudioapi::getSourceEditorContext(), which
# returns NULL outside the RStudio editor, so it could only ever be run by hand.

library(optparse)
library(dplyr)
library(readr)
library(tidyr)
library(purrr)
library(stringr)   # str_split_fixed below; previously used without being loaded

option_list <- list(
  make_option("--knn-raw", type = "character", dest = "knn_raw",
              help = "Directory of LOO sweep CSVs, searched recursively. Each
                file's parent directory name is taken as the held-out serotype."),
  make_option("--ground-truth", type = "character", dest = "ground_truth",
              help = "CSV of sample_id, Serotype, Serogroup, dataset, benchmark."),
  make_option("--output", type = "character", dest = "output",
              help = "Directory for the k{K}_knn_data.csv outputs.")
)
opt <- parse_args(OptionParser(option_list = option_list))

for (required in c("knn_raw", "ground_truth", "output")) {
  if (is.null(opt[[required]])) {
    stop(sprintf("--%s is required", gsub("_", "-", required)), call. = FALSE)
  }
}
if (!dir.exists(opt$knn_raw)) stop("--knn-raw directory not found: ", opt$knn_raw, call. = FALSE)
if (!file.exists(opt$ground_truth)) stop("--ground-truth not found: ", opt$ground_truth, call. = FALSE)

data_root <- opt$output
dir.create(data_root, recursive = TRUE, showWarnings = FALSE)

ground_truth <- read.csv(opt$ground_truth)
colnames(ground_truth) <- c("sample_id", "Serotype", "Serogroup", "dataset", "benchmark")

plot_dir <- file.path(data_root, "plots")
dir.create(plot_dir, showWarnings = FALSE)

# create file containing all knns
process_knn_raw <- function(file) {
  df <- read_csv(file, show_col_types = FALSE)
  
  file_str_list <- strsplit(file, "/")[[1]]
  loo_serotype <- file_str_list[length(file_str_list) - 1]
  
  sample_id_new <- str_split_fixed(df$sample_id, "#", 2)
  df$sample_id <- sample_id_new[,1]
  df$Contig_ID <- sample_id_new[,2]
  
  nn_sample_id_new <- str_split_fixed(df$nn_sample_id, "\\|", 2)
  df$nn_type <- nn_sample_id_new[,1]
  nn_sample_id_new <- str_split_fixed(nn_sample_id_new[,2], "#", 2)
  df$nn_sample_id <- nn_sample_id_new[,1]
  df$nn_Contig_ID <- nn_sample_id_new[,2]
  
  df$Contig_ID <- as.character(df$Contig_ID)
  df$nn_Contig_ID <- as.character(df$nn_Contig_ID)
  df$nn_genogroup <- as.character(df$nn_genogroup)
  
  df$loo_serotype <- loo_serotype
  df$nn_serotype <- as.character(df$nn_serotype)
  
  df <- df %>%
    mutate(
      loo_serotype = trimws(sub("(?i)serogroup\\s*", "", loo_serotype, perl = TRUE)),
      loo_serotype = sub("^0+([0-9])", "\\1", loo_serotype),
      loo_serogroup = sub("^([0-9]+).*", "\\1", loo_serotype),
      nn_serotype = trimws(sub("(?i)serogroup\\s*", "", nn_serotype, perl = TRUE)),
      nn_serotype =sub("^0+([0-9])", "\\1", nn_serotype),
      nn_serogroup = sub("^([0-9]+).*", "\\1", nn_serotype),
      nn_genogroup = trimws(sub("(?i)serogroup\\s*", "", nn_genogroup, perl = TRUE))
    )
  
  df <- df |>
    select(k, sample_id, Contig_ID, loo_serotype, loo_serogroup, knn_distance, nn_serotype, nn_serogroup, nn_genogroup, nn_sample_id, nn_Contig_ID, nn_type)
  df
}

# create merged file of knn distances at K=1
# `.` in the pattern is a regex wildcard here; harmless, kept as-is for behaviour.
files <- list.files(opt$knn_raw, pattern = "knn_query_distances_kgrid.csv", recursive = TRUE, full.names = TRUE)
all_results <- lapply(files, process_knn_raw)
combined_query <- bind_rows(all_results)
combined_query$is_held_out <- TRUE

files <- list.files(opt$knn_raw, pattern = "knn_id_distances_kgrid.csv", recursive = TRUE, full.names = TRUE)
all_results <- lapply(files, process_knn_raw)
combined_training <- bind_rows(all_results)
combined_training$is_held_out <- FALSE

combined <- rbind(combined_query, combined_training)

# write files per-k value
k_vals <- sort(unique(combined$k))

for (k_val in k_vals) {
  k_df <- subset(combined, k == k_val)
  combined_merged <- left_join(k_df, ground_truth, by = c("sample_id"))
  
  # merge with serotype information
  combined_merged <- combined_merged %>%
    mutate(
      Serotype = trimws(sub("(?i)serogroup\\s*", "", Serotype, perl = TRUE)),
      Serotype = sub("^0+([0-9])", "\\1", Serotype),
      Serogroup = sub("^([0-9]+).*", "\\1", Serotype),
    )
  
  # determine with genogroup assignments
  genogroups <- combined_merged %>%
    distinct(nn_serotype, nn_serogroup, nn_genogroup)
  colnames(genogroups) <- c("Serotype", "Serogroup", "Genogroup")
  
  # merge genogroups
  final_knn_df <- combined_merged %>%
    left_join(
      genogroups %>%
        group_by(Serotype) %>%
        slice(1) %>%
        ungroup() %>%
        select(Serotype, Genogroup),
      by = "Serotype"
    )
  
  write_csv(final_knn_df, file.path(data_root, paste0("k", k_val, "_knn_data.csv")))
}

# Optionally restrict to the serotypes ALLCAPS can actually predict. Left off: the
# list was read from a hand-staged ALLCAPS_possible_serotypes.csv that no stage
# produces, so requiring it would break the command for everyone else.
# final_knn_df <- final_knn_df %>% filter(loo_serotype %in% allcaps_serotypes)

message(sprintf("Wrote %d k-value dataset(s) to %s", length(k_vals), data_root))
