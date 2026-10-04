library(dplyr)
library(readr)
library(tidyr)
library(purrr)
library(Polychrome)
library(ggplot2)
library(ggpubr)
library(stringr)
library(ggsci)
library(ggpubr)
library(MLmetrics)
library(pROC)

data_root <- file.path(dirname(rstudioapi::getSourceEditorContext()$path), "data")
ground_truth <- read.csv(file.path(data_root, "merged_ground_truth.csv"))
colnames(ground_truth) <- c("sample_id", "Serotype", "Serogroup", "dataset", "benchmark")

# assign out put directories
plot_dir <- file.path(data_root, "plots")
dir.create(plot_dir, showWarnings = FALSE)

files <- list.files(file.path(data_root), pattern = "k*_knn_data.csv", recursive = TRUE, full.names = TRUE)

# calculating per serotype novel classification accuracy
match_serotype  <- function(vec, cls) !is.na(vec) & vec == cls
compute_metrics <- function(final_knn_df, pred_col_name, f1_beta) {
  classes <- sort(unique(final_knn_df$loo_serotype))
  map_dfr(classes, function(cls) {
    class_rows <- final_knn_df[match_serotype(final_knn_df$loo_serotype, cls),]
    true_col <- as.logical(class_rows$is_held_out)
    pred_col <- as.logical(class_rows[[pred_col_name]])
    
    TP <- sum(true_col & pred_col, na.rm = TRUE)
    FP <- sum(!true_col & pred_col, na.rm = TRUE)
    FN <- sum( true_col & !pred_col, na.rm = TRUE)
    TN <- sum(!true_col & !pred_col, na.rm = TRUE)
    Total <- TP + FN
    
    f1 <- F1_Score(y_true = true_col, y_pred = pred_col, positive = TRUE)
    sensitivity <- Sensitivity(y_true = true_col, y_pred = pred_col, positive = TRUE)
    specificity <- Specificity(y_true = true_col, y_pred = pred_col, positive = TRUE)
    accuracy <- Accuracy(y_true = true_col, y_pred = pred_col)
    precision <- Precision(y_true = true_col, y_pred = pred_col, positive = TRUE)
    
    weighted_f1 <- (1 + f1_beta^2) * (precision * sensitivity) /
      (f1_beta^2 * precision + sensitivity)
    
    row <- tibble(Serotype = as.character(cls), TP, FP, FN, TN, Total, sensitivity, specificity, precision, accuracy, f1, weighted_f1)
    row <- row %>% replace(is.na(.), 0.0)
    row
  })
}

compute_distances <- function(final_knn_df, pred_col_name) {
  classes <- sort(unique(final_knn_df$loo_serotype))
  map_dfr(classes, function(cls) {
    
    class_rows <- final_knn_df[match_serotype(final_knn_df$loo_serotype, cls),]
    true_col <- as.logical(class_rows$is_held_out)
    pred_col <- as.logical(class_rows[[pred_col_name]])
    
    distances_TP <- class_rows[true_col & pred_col,]
    distances_FP <- class_rows[!true_col & pred_col,]
    distances_FN <- class_rows[true_col & !pred_col,]
    distances_TN <- class_rows[!true_col & !pred_col,]
    
    distances_held_out <- class_rows[true_col,]
    distances_training <- class_rows[!true_col,]
    
    quantiles_distances_TP <-  quantile(distances_TP$knn_distance, c(0.25, 0.5, 0.75))
    quantiles_distances_FP <-  quantile(distances_FP$knn_distance, c(0.25, 0.5, 0.75))
    quantiles_distances_FN <-  quantile(distances_FN$knn_distance, c(0.25, 0.5, 0.75))
    quantiles_distances_TN <-  quantile(distances_TN$knn_distance, c(0.25, 0.5, 0.75))
    
    quantiles_distances_held_out <- quantile(distances_held_out$knn_distance, c(0.25, 0.5, 0.75))
    quantiles_distances_training <- quantile(distances_training$knn_distance, c(0.25, 0.5, 0.75))
    
    row <- tibble(Serotype = as.character(cls),  
                  LQ_distances_TP = quantiles_distances_TP[1], median_distances_TP = quantiles_distances_TP[2], UQ_distances_TP = quantiles_distances_TP[3],
                  LQ_distances_FP = quantiles_distances_FP[1], median_distances_FP = quantiles_distances_FP[2], UQ_distances_FP = quantiles_distances_FP[3],
                  LQ_distances_TN = quantiles_distances_TN[1], median_distances_TN = quantiles_distances_TN[2], UQ_distances_TN = quantiles_distances_TN[3],
                  LQ_distances_FN = quantiles_distances_FN[1], median_distances_FN = quantiles_distances_FN[2], UQ_distances_FN = quantiles_distances_FN[3],
                  LQ_distances_held_out = quantiles_distances_held_out[1], median_distances_held_out = quantiles_distances_held_out[2], UQ_distances_held_out = quantiles_distances_held_out[3],
                  LQ_distances_training = quantiles_distances_training[1], median_distances_training = quantiles_distances_training[2], UQ_distances_training = quantiles_distances_training[3])
    row <- row %>% replace(is.na(.), 0.0)
    row
  })
}

# read in first file, decide on samples to be testing and training to be consistent across k values
set.seed(42)
first_file <- files[1]
samples <- read_csv(first_file, show_col_types = FALSE)

# testing_data <- example_knn_df %>%
#   filter(dataset == "Testing" & benchmark == "GPS benchmark")
# training_data <- example_knn_df %>%
#   filter(dataset == "Training" & benchmark == "GPS benchmark")

# only include genomes used in training
samples <- samples %>%
  filter(dataset == "Training" & benchmark == "GPS benchmark")
samples$idx <- paste0(samples$sample_id, "_", samples$Contig_ID, "_", samples$loo_serotype, "_",samples$nn_serotype, "_", samples$is_held_out)

# Select 90% of samples within each nn_serotype x is_held_out group
train_samples <- samples %>%
  group_by(nn_serotype, is_held_out) %>%
  slice_sample(prop = 0.9) %>%
  ungroup()
train_samples$dataset <- "Training"

test_samples <- samples %>%
  filter(!idx %in% train_samples$idx)
test_samples$dataset <- "Testing"

all_samples <- rbind(train_samples, test_samples)
write_csv(all_samples, file.path(data_root, paste0("nn_training_testing_data.csv")))

for (file in files) {
  final_knn_df <- read_csv(file, show_col_types = FALSE)
  final_knn_df$idx <- paste0(final_knn_df$sample_id, "_", final_knn_df$Contig_ID, "_", final_knn_df$loo_serotype, "_",final_knn_df$nn_serotype, "_", final_knn_df$is_held_out)
  
  k_val <- unique(final_knn_df$k)
  
  final_knn_df <- final_knn_df %>%
    mutate(.row_id = row_number())
  
  # split train and test data and check
  train <- final_knn_df %>%
    filter(idx %in% train_samples$idx)
  
  test <- final_knn_df %>%
    filter(idx %in% test_samples$idx)
  
  # generate ROC curves per serotype
  roc_results <- train %>%
    group_by(nn_serotype) %>%
    filter(
      n_distinct(is_held_out) == 2,
      !is.na(knn_distance)
    ) %>%
    nest() %>%
    mutate(
      roc = map(
        data,
        ~ roc(
          response = .x$is_held_out,
          predictor = .x$knn_distance,
          levels = c(FALSE, TRUE),
          direction = "<",
          quiet = TRUE
        )
      ),
      auc = map_dbl(roc, auc)
    )
  
  # pick best threshold per serotype
  best_thresholds <- roc_results %>%
    mutate(
      best = purrr::map(
        roc,
        ~ pROC::coords(
          .x,
          x = "best",
          best.method = "youden",
          ret = c("threshold", "sensitivity", "specificity"),
          transpose = FALSE
        )
      )
    ) %>%
    select(nn_serotype, auc, best) %>%
    tidyr::unnest(best)
  
  best_thresholds$k <- k_val
  
  write_csv(best_thresholds, file.path(data_root, paste0(k_val, "_nn_best_thresholds_per_serotype.csv")))
  
  # generate ROC curves for all
  roc_result_all <- train %>%
    nest() %>%
    mutate(
      roc = map(
        data,
        ~ roc(
          response = .x$is_held_out,
          predictor = .x$knn_distance,
          levels = c(FALSE, TRUE),
          direction = "<",
          quiet = TRUE
        )
      ),
      auc = map_dbl(roc, auc)
    )
  
  # pick best threshold for all
  best_threshold_all <- roc_result_all %>%
    mutate(
      best = purrr::map(
        roc,
        ~ pROC::coords(
          .x,
          x = "best",
          best.method = "youden",
          ret = c("threshold", "sensitivity", "specificity"),
          transpose = FALSE
        )
      )
    ) %>%
    select(auc, best) %>%
    tidyr::unnest(best)
  
  best_threshold_all$k <- k_val
  
  write_csv(best_threshold_all, file.path(data_root, paste0(k_val, "_nn_best_thresholds_all.csv")))
  
  # classify each serotype based on ROC
  final_knn_df <- final_knn_df %>%
    left_join(
      best_thresholds %>%
        select(nn_serotype, threshold),
      by = "nn_serotype"
    ) %>%
    mutate(
      is_novel_threshold_per_serotype = if_else(
        is.na(threshold),
        FALSE,
        knn_distance > threshold
      )
    )
  
  # classify test data
  test <- test %>%
    left_join(
      best_thresholds %>%
        select(nn_serotype, threshold),
      by = "nn_serotype"
    ) %>%
    mutate(
      is_novel_threshold_per_serotype = if_else(
        is.na(threshold),
        FALSE,
        knn_distance > threshold
      )
    )
  
  final_knn_df$threshold_all <- best_threshold_all$threshold
  test$threshold_all <- best_threshold_all$threshold
  
  final_knn_df <- final_knn_df %>%
    mutate(
      is_novel_threshold_all = if_else(
        is.na(threshold),
        FALSE,
        knn_distance > threshold_all
      )
    )
  
  # classify test data
  test <- test %>%
    mutate(
      is_novel_threshold_all = if_else(
        is.na(threshold),
        FALSE,
        knn_distance > threshold_all
      )
    )
  
  # determine classification accuracy
  f1_beta <- 4
  knn_metrics_per_serotype <- compute_metrics(test, "is_novel_threshold_per_serotype", f1_beta)
  knn_metrics_per_serotype$k <- k_val
  knn_metrics_all <- compute_metrics(test, "is_novel_threshold_all", f1_beta)
  knn_metrics_all$k <- k_val
  write_csv(knn_metrics_per_serotype, file.path(data_root, paste0("all_", k_val, "_nn_metrics_threshold_per_serotype.csv")))
  write_csv(knn_metrics_all, file.path(data_root, paste0("all_", k_val, "_nn_metrics_threshold_all.csv")))
  
  write_csv(final_knn_df, file.path(data_root, paste0(k_val, "_nn_novelty_classification_results.csv")))
  
  # determine distances of TPs etc
  knn_distances_per_serotype <- compute_distances(final_knn_df, "is_novel_threshold_per_serotype")
  knn_distances_all <- compute_distances(final_knn_df, "is_novel_threshold_all")
  
  # per serotype
  plot_data <- knn_distances_per_serotype %>%
    select(Serotype,
           LQ_distances_TP,
           median_distances_TP,
           UQ_distances_TP,
           LQ_distances_FP,
           median_distances_FP,
           UQ_distances_FP,
           LQ_distances_TN,
           median_distances_TN,
           UQ_distances_TN,
           LQ_distances_FN,
           median_distances_FN,
           UQ_distances_FN) %>%
    pivot_longer(
      cols = c(
        LQ_distances_TP,
        median_distances_TP,
        UQ_distances_TP,
        LQ_distances_FP,
        median_distances_FP,
        UQ_distances_FP,
        LQ_distances_TN,
        median_distances_TN,
        UQ_distances_TN,
        LQ_distances_FN,
        median_distances_FN,
        UQ_distances_FN
      ),
      names_to = c("statistic", "group"),
      names_pattern = "^(LQ|median|UQ)_distances_(TP|FP|TN|FN)$"
    ) %>%
    pivot_wider(
      names_from = statistic,
      values_from = value
    ) %>%
    rename(
      ymin = LQ,
      middle = median,
      ymax = UQ
    )
  
  #plot_data$group[plot_data$group == "held_out"] <- "Novel"
  #plot_data$group[plot_data$group == "training"] <- "Training"
  
  plot_data <- plot_data %>%
    mutate(
      serotype_number = as.numeric(sub("^([0-9]+).*", "\\1", Serotype)),
      serotype_suffix = sub("^[0-9]+", "", Serotype)
    ) %>%
    arrange(serotype_number, serotype_suffix) %>%
    mutate(
      Serotype = factor(Serotype, levels = unique(Serotype))
    ) %>%
    select(-serotype_number, -serotype_suffix)
  
  plot_data$group <- factor(plot_data$group, levels = c("TP", "FP", "TN", "FN"))
  
  # generate distance plots showing distances between NNs for held-out and known serotypes
  p.distance <- ggplot(
    plot_data,
    aes(
      x = Serotype,
      y = middle,
      ymin = ymin,
      ymax = ymax,
      group = group
    )
  ) +
    geom_errorbar(
      aes(color = group),
      position = position_dodge(width = 0.5),
      width = 0.15
    ) +
    geom_point(
      aes(color = group),
      position = position_dodge(width = 0.5),
      size = 2
    ) +
    # Per-serotype threshold
    geom_errorbar(
      data = best_thresholds,
      aes(
        x = nn_serotype,
        ymin = threshold,
        ymax = threshold
      ),
      width = 0.7,
      color = "purple",
      linewidth = 1,
      inherit.aes = FALSE
    ) +
    coord_flip() +
    labs(
      x = "Serotype",
      y = "1st NN Cosine Distance",
      color = "Classification"
    ) +
    theme_light() +
    scale_color_npg()
  p.distance
  
  ggsave(file.path(plot_dir, paste0(k_val, "_nn_distance_quartiles_threshold_per_serotype.png")), plot=p.distance, width = 9, height = 11)
  ggsave(file.path(plot_dir, paste0(k_val, "_nn_distance_quartiles_threshold_per_serotype.pdf")), plot=p.distance, width = 9, height = 11)
  
  # all threshold
  plot_data <- knn_distances_all %>%
    select(Serotype,
           LQ_distances_TP,
           median_distances_TP,
           UQ_distances_TP,
           LQ_distances_FP,
           median_distances_FP,
           UQ_distances_FP,
           LQ_distances_TN,
           median_distances_TN,
           UQ_distances_TN,
           LQ_distances_FN,
           median_distances_FN,
           UQ_distances_FN) %>%
    pivot_longer(
      cols = c(
        LQ_distances_TP,
        median_distances_TP,
        UQ_distances_TP,
        LQ_distances_FP,
        median_distances_FP,
        UQ_distances_FP,
        LQ_distances_TN,
        median_distances_TN,
        UQ_distances_TN,
        LQ_distances_FN,
        median_distances_FN,
        UQ_distances_FN
      ),
      names_to = c("statistic", "group"),
      names_pattern = "^(LQ|median|UQ)_distances_(TP|FP|TN|FN)$"
    ) %>%
    pivot_wider(
      names_from = statistic,
      values_from = value
    ) %>%
    rename(
      ymin = LQ,
      middle = median,
      ymax = UQ
    )
  
  #plot_data$group[plot_data$group == "held_out"] <- "Novel"
  #plot_data$group[plot_data$group == "training"] <- "Training"
  
  plot_data <- plot_data %>%
    mutate(
      serotype_number = as.numeric(sub("^([0-9]+).*", "\\1", Serotype)),
      serotype_suffix = sub("^[0-9]+", "", Serotype)
    ) %>%
    arrange(serotype_number, serotype_suffix) %>%
    mutate(
      Serotype = factor(Serotype, levels = unique(Serotype))
    ) %>%
    select(-serotype_number, -serotype_suffix)
  
  plot_data$group <- factor(plot_data$group, levels = c("TP", "FP", "TN", "FN"))
  
  # generate distance plots showing distances between NNs for held-out and known serotypes
  overall_threshold <- best_threshold_all$threshold
  p.distance <- ggplot(
    plot_data,
    aes(
      x = Serotype,
      y = middle,
      ymin = ymin,
      ymax = ymax,
      group = group
    )
  ) +
    geom_errorbar(
      aes(color = group),
      position = position_dodge(width = 0.5),
      width = 0.15
    ) +
    geom_point(
      aes(color = group),
      position = position_dodge(width = 0.5),
      size = 2
    ) +
    coord_flip() +
    # overall
    geom_hline(
      yintercept = overall_threshold,
      color = "purple",
      linewidth = 1,
      linetype = "dashed"
    ) +
    labs(
      x = "Serotype",
      y = "1st NN Cosine Distance",
      color = "Sample Type"
    ) +
    theme_light() +
    scale_color_npg()
  p.distance
  
  ggsave(file.path(plot_dir, paste0(k_val, "_nn_distance_quartiles_threshold_all.png")), plot=p.distance, width = 9, height = 11)
  ggsave(file.path(plot_dir, paste0(k_val, "_nn_distance_quartiles_threshold_all.png")), plot=p.distance, width = 9, height = 11)
}
