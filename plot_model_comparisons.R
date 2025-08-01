library(tidyverse)
library(ggplot2)
library(patchwork)

# Add these lines to the top of your R script:
args <- commandArgs(trailingOnly = TRUE)
grouping_var <- args[1]
outcome_var <- args[2]
data_type <- args[3]

# Function to read and tag model info
read_metrics_file <- function(filename) {
  base_name <- basename(filename)
  cat("Processing file:", base_name, "\n")
  
  if (grepl("^augmented_xlearn_", base_name)) {
    model_type <- "augmented_xlearn"
  } else if (grepl("^grouped_xlearn_", base_name)) {
    model_type <- "grouped_xlearn"
  } else if (grepl("^local_", base_name)) {
    model_type <- "local"
  } else if (grepl("^global_", base_name)) {
    model_type <- "global"
  } else if (grepl("^xlearn_", base_name)) {
    model_type <- "xlearn"
  }
  
  
  cat("  Identified model type:", model_type, "\n")
  
  if (grepl("grouped_xlearn", base_name)) {
    arch_pattern <- "grouped_xlearn_(tree|bart|ranger|regression)_"
  } else {
    arch_pattern <- "_(tree|bart|ranger|regression)"
  }
  
  arch_match <- str_match(base_name, arch_pattern)
  if (is.na(arch_match[1])) {
    cat("  No architecture match found\n")
    return(NULL)
  }
  architecture <- arch_match[2]
  cat("  Identified architecture:", architecture, "\n")
  
  read_csv(filename, show_col_types = FALSE) %>%
    mutate(model_type = model_type, architecture = architecture)
}

# Load all data
metrics_files <- list.files(file.path("results/model_comparison", data_type),
                            pattern = "*_metrics.csv",
                            full.names = TRUE)

cat("\nFound", length(metrics_files), "metrics files\n")
all_metrics <- map_df(metrics_files, read_metrics_file)

# Model aesthetics
model_order <- c("local", "global", "xlearn",
                 "grouped_xlearn", "augmented_xlearn")

model_colors <- c(
  "local" = "#003f5c", "global" = "#58508d", "xlearn" = "#bc5090",
  "grouped_xlearn" = "#ff6361", "augmented_xlearn" = "#ffa600"
)

model_labels <- c(
  "local" = "Local", "global" = "Global", "xlearn" = "X-Learn",
  "grouped_xlearn" = "Grouped X-Learn",
  "augmented_xlearn" = "Augmented X-Learn"
)

create_metric_plot <- function(data, metric_name, tag_label) {
  data <- data %>% mutate(model_type = factor(model_type, levels = model_order))
  plot_data <- data %>% filter(metric == metric_name)
  if (nrow(plot_data) == 0) return(NULL)
  
  # Compute IQR-based outlier threshold
  q1 <- quantile(plot_data$value, 0.25, na.rm = TRUE)
  q3 <- quantile(plot_data$value, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  upper_whisker <- q3 + 1.5 * iqr
  
  # Mark outliers
  plot_data <- plot_data %>%
    mutate(
      is_outlier = value > upper_whisker
    )
  
  # Set y-axis limit based on inliers only
  max_inlier <- max(plot_data$value[!plot_data$is_outlier], na.rm = TRUE)
  y_limit <- max_inlier * 1.05
  y_lower <- min(plot_data$value, na.rm = TRUE) * 0.95
  
  # Plot
  ggplot(plot_data, aes(x = architecture, y = value, fill = model_type)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.9)) +
    scale_fill_manual(values = model_colors,
                      name = "Model Structure",
                      breaks = model_order,
                      labels = model_labels) +
    scale_x_discrete(
      limits = c("tree", "bart", "ranger", "regression"),
      labels = c("Tree", "Bart", "Ranger", "Regression")
    ) +
    scale_y_continuous(expand = c(0, 0)) +
    coord_cartesian(ylim = c(y_lower, y_limit)) +
    labs(title = NULL, y = NULL, tag = tag_label) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 11, color = "#444444"),
      axis.text.y = element_text(size = 11, color = "#444444"),
      axis.title.x = element_blank(),
      axis.title.y = element_blank(),
      legend.title = element_text(size = 10),
      legend.text = element_text(size = 9),
      plot.title = element_text(hjust = 0.5, size = 14),
      legend.position = "bottom",
      legend.box = "horizontal"
    ) +
    guides(fill = guide_legend(nrow = 2))
}


# Output folder
output_dir <- file.path("plots/", data_type)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Metric and tag structure
grouped_metrics <- list(
  mse = list(metrics = c("mse", "small_mse"), tags = c("(a)", "(b)")),
  resolution = list(metrics = c("resolution", "small_resolution"), tags = c("(c)", "(d)")),
  calibration = list(metrics = c("calibration", "small_calibration"), tags = c("(e)", "(f)"))
)

# Pretty names for titles
display_titles <- c(
  mse = "MSE", small_mse = "Small MSE",
  resolution = "Resolution", small_resolution = "Small Resolution",
  calibration = "Calibration Error", small_calibration = "Small Calibration Error"
)

# Build individual plots and combine columns
metric_list <- unlist(map(grouped_metrics, "metrics"))
tag_list <- unlist(map(grouped_metrics, "tags"))

named_plots <- map2(metric_list, tag_list, ~create_metric_plot(all_metrics, .x, .y) + ggtitle(display_titles[[.x]]))
names(named_plots) <- metric_list

# Arrange each pair (top full, bottom small) in a column
column1 <- wrap_plots(named_plots$mse, named_plots$small_mse, ncol = 1)
column2 <- wrap_plots(named_plots$resolution, named_plots$small_resolution, ncol = 1)
column3 <- wrap_plots(named_plots$calibration, named_plots$small_calibration, ncol = 1)

# Combine horizontally
combined_all <- wrap_plots(column1, column2, column3, ncol = 3, guides = "collect") +
  plot_annotation(
    title = paste0("Model Performance – ", str_to_title(grouping_var)," Groups"),
    theme = theme(plot.title = element_text(size = 18, hjust = 0.5))
  ) &
  theme(
    legend.position = "bottom",
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 9)
  )

# Save
ggsave(
  filename = file.path(output_dir, "model_performance_metrics.png"),
  plot = combined_all,
  width = 16,
  height = 12
)
