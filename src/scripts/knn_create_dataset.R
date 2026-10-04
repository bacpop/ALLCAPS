library(dplyr)
library(readr)
library(tidyr)
library(purrr)

data_root <- file.path(dirname(rstudioapi::getSourceEditorContext()$path), "data")
ground_truth <- read.csv(file.path(data_root, "merged_ground_truth.csv"))
colnames(ground_truth) <- c("sample_id", "Serotype", "Serogroup", "dataset", "benchmark")

# sample serotypes predicted by ALLCAPS
allcaps_serotypes <- read.csv(
  file.path(dirname(rstudioapi::getSourceEditorContext()$path),
            "data", "ALLCAPS_possible_serotypes.csv")
)$Serotypes

# assign out put directories
plot_dir <- file.path(data_root, "plots")
dir.create(plot_dir, showWarnings = FALSE)
create_merged_knn_file <- FALSE

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
files <- list.files(file.path(data_root, "knn-sweep", "knn_raw"), pattern = "knn_query_distances_kgrid.csv", recursive = TRUE, full.names = TRUE)
all_results <- lapply(files, process_knn_raw)
combined_query <- bind_rows(all_results)
combined_query$is_held_out <- TRUE

files <- list.files(file.path(data_root, "knn-sweep", "knn_raw"), pattern = "knn_id_distances_kgrid.csv", recursive = TRUE, full.names = TRUE)
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

# filter combined merged
# final_knn_df <- final_knn_df %>% filter(loo_serotype %in% allcaps_serotypes)
