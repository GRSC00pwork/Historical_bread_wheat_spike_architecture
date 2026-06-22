#******************Descriptive Stats (Table S# AND Figure S1)*******************
library(dplyr)
library(tidyr)
library(openxlsx)
library(ggplot2)
library(patchwork)

desc_stats <- df %>%
  select(all_of(c(cultivar_col, trait_cols))) %>%
  pivot_longer(cols = all_of(trait_cols), names_to = "trait", values_to = "value") %>%
  group_by(!!sym(cultivar_col), trait) %>%
  summarise(
    n      = sum(!is.na(value)),
    mean   = round(mean(value, na.rm = TRUE), 4),
    sd     = round(sd(value,   na.rm = TRUE), 4),
    min    = round(min(value,  na.rm = TRUE), 4),
    max    = round(max(value,  na.rm = TRUE), 4),
    se     = round(sd / sqrt(n), 4),
    cv_pct = round(ifelse(mean != 0, (sd / abs(mean)) * 100, NA_real_), 2),
    .groups = "drop"
  ) %>%
  arrange(trait, !!sym(cultivar_col))
#ONE-WAY ANOVA + H2 
sig_star <- function(p) {
  if (is.na(p)) return("")
  if (p < 0.001) return("***")
  if (p < 0.01)  return("**")
  if (p < 0.05)  return("*")
  if (p < 0.1)   return(".")
  return("ns")
}
anova_rows <- list()
for (trait in trait_cols) {
  vals_clean <- df[[trait]][!is.na(df[[trait]])]
  if (length(vals_clean) < 10 || sd(vals_clean) == 0) {
    cat(sprintf("  %-45s  SKIPPED (insufficient data)\n", trait))
    next
  }
  fit <- tryCatch(
    aov(as.formula(paste0("`", trait, "` ~ ", cultivar_col)), data = df),
    error = function(e) { cat(sprintf("  %-45s  ERROR: %s\n", trait, e$message)); NULL }
  )
  if (is.null(fit)) next
  
  at   <- anova(fit)
  Df_c <- at[cultivar_col, "Df"];      SS_c <- at[cultivar_col, "Sum Sq"]
  MS_c <- at[cultivar_col, "Mean Sq"]; F_c  <- at[cultivar_col, "F value"]
  p_c  <- at[cultivar_col, "Pr(>F)"]
  Df_e <- at["Residuals",  "Df"];      SS_e <- at["Residuals",  "Sum Sq"]
  MS_e <- at["Residuals",  "Mean Sq"]
  
  sigma2_G <- max((MS_c - MS_e) / EXPECTED_REPS, 0)
  sigma2_E <- MS_e
  denom    <- sigma2_G + sigma2_E / EXPECTED_REPS
  H2       <- if (denom > 0) max(min(sigma2_G / denom, 1), 0) else 0
  grand_mean <- mean(vals_clean)
  cv_pct     <- if (grand_mean != 0) (sqrt(MS_e) / abs(grand_mean)) * 100 else NA_real_
  anova_rows[[trait]] <- data.frame(
    Trait        = trait,
    Grand_Mean   = round(grand_mean, 4),
    CV_percent   = round(cv_pct, 2),
    Df_Cultivar  = Df_c,
    SS_Cultivar  = round(SS_c, 4),
    MS_Cultivar  = round(MS_c, 4),
    F_value      = round(F_c, 3),
    p_value      = signif(p_c, 4),
    Significance = sig_star(p_c),
    Df_Error     = Df_e,
    SS_Error     = round(SS_e, 4),
    MS_Error     = round(MS_e, 4),
    Sigma2_G     = round(sigma2_G, 6),
    Sigma2_E     = round(sigma2_E, 6),
    H2_broad     = round(H2, 4),
    stringsAsFactors = FALSE
  )}
anova_table <- bind_rows(anova_rows)
#SUMMARY 
n_traits <- nrow(anova_table)
n_high   <- sum(anova_table$H2_broad >  0.70)
n_med    <- sum(anova_table$H2_broad >= 0.40 & anova_table$H2_broad <= 0.70)
n_low    <- sum(anova_table$H2_broad <  0.40)
sig_tbl  <- table(factor(anova_table$Significance,
                         levels = c("***", "**", "*", ".", "ns")))
summary_sheet <- data.frame(
  Metric = c("Cultivars", "Reps per cultivar", "Traits analyzed",
             "Sig ***", "Sig **", "Sig *", "Sig .", "Sig ns",
             "H2 > 0.70", "H2 0.40-0.70", "H2 < 0.40"),
  Value  = c(nlevels(df[[cultivar_col]]), EXPECTED_REPS, n_traits,
             sig_tbl["***"], sig_tbl["**"], sig_tbl["*"], sig_tbl["."], sig_tbl["ns"],
             n_high, n_med, n_low))
#COMBINED LOLLIPOP PLOT: (A) F-value, (B) H2
top_n <- min(40, nrow(anova_table))
plot_df <- heritability %>% head(top_n) %>%
  mutate(Trait = factor(Trait, levels = rev(Trait)))

# Panel A: F-value
pA <- ggplot(plot_df, aes(x = Trait, y = F_value)) +
  geom_segment(aes(xend = Trait, y = 0, yend = F_value),
               color = "gray70", linewidth = 0.5) +
  geom_point(color = "#0078BD", size = 2) +
  geom_text(aes(label = paste0(sprintf("%.1f", F_value), Significance)),
            hjust = -0.25, size = 2.8, fontface = "bold",
            color = "black") +
  coord_flip() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.28))) +
  labs(title = "A",
       x = NULL, y = "F-value") +
  theme_minimal(base_size = 10) +
  theme(axis.line = element_line(color = "black", linewidth = 0.5),
        axis.ticks = element_line(color = "black", linewidth = 0.5),
        plot.title = element_text(face = "bold", hjust = -0.15),
        axis.title.x = element_text(size = 10, face = "bold", color = "black"),
        axis.text.y = element_text(size = 8, face = "bold", color = "black"),
        axis.text.x = element_text(size = 8, face = "bold", color = "black"),
        panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.minor   = element_blank(),
        legend.position = "none")

# Panel B: H2
pB <- ggplot(plot_df, aes(x = Trait, y = H2_broad)) +
  geom_segment(aes(xend = Trait, y = 0, yend = H2_broad),
               color = "gray70", linewidth = 0.5) +
  geom_point(color = "#E42237", size = 2) +
  geom_text(aes(label = sprintf("%.2f", H2_broad)), hjust = -0.8, size = 2.6) +
  coord_flip() +
  scale_y_continuous(limits = c(0, 1.1), breaks = seq(0, 1, 0.2),
                     expand = expansion(mult = c(0, 0))) +
  labs(title = expression(bold("B")),
       x = NULL, y = expression(paste("Heritability (", H^2, ")"))) +
  theme_minimal(base_size = 10) +
  theme(axis.line = element_line(color = "black", linewidth = 0.5),
        axis.ticks = element_line(color = "black", linewidth = 0.5),
        plot.title = element_text(face = "bold", hjust = -0.15),
        axis.text.y = element_text(size = 8, face = "bold", color = "black"),
        axis.title.x = element_text(size = 10, face = "bold", color = "black"),
        axis.text.x = element_text(size = 8, face = "bold", color = "black"),
        panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.minor   = element_blank(),
        legend.position = "none")

combined <- pA + pB + plot_layout(ncol = 2)

ggsave(output_plot, combined,
       width = 8, height = max(4, top_n * 0.25), dpi = 600)

#*******************************************************************************
#*****************Figures 1A, 1B, 1C, 1D****************************************

library(readxl)
library(dplyr)
library(ggplot2)
library(patchwork)

traits <- c("SL_mm", "TFSk", "TFiSk", "TSkS")

trait_labels <- c(
  SL_mm = "A",
  TFSk  = "B",
  TFiSk = "C",
  TSkS  = "D"
)

legend_names <- c(
  SL_mm = "SL (mm)",
  TFSk  = "TFSk",
  TFiSk = "TFiSk",
  TSkS  = "TSkS"
)

axis_labels <- c(
  SL_mm = "Manually measured SL (mm)",
  TFSk  = "Manually counted TFSk",
  TFiSk = "Manually counted TFiSk",
  TSkS  = "Manually counted TSkS"
)

y_axis_labels <- c(
  SL_mm = "Image predicted SL (mm)",
  TFSk  = "Image predicted TFSk",
  TFiSk = "Image predicted TFiSk",
  TSkS  = "Image predicted TSkS"
)

#Compute Stats
compute_stats <- function(x, y) {
  fit       <- lm(y ~ x)
  r         <- cor(x, y, use = "complete.obs")
  r2        <- r^2
  rmse      <- sqrt(mean((y - x)^2, na.rm = TRUE))
  intercept <- coef(fit)[1]
  slope     <- coef(fit)[2]
  list(r = r, r2 = r2, rmse = rmse, intercept = intercept, slope = slope)
}

#Build Individual Plots
plot_list <- list()
for (trait in traits) {
  x <- data[[paste0(trait, "_meas")]]
  y <- data[[paste0(trait, "_pred")]]
  
  s   <- compute_stats(x, y)
  lbl <- trait_labels[[trait]]
  
  ann <- paste0(
    "Intercept = ", round(s$intercept, 2), "\n",
    "R = ",         round(s$r, 2), "\n",
    "R² = ",        round(s$r2, 2), "\n",
    "RMSE = ",      round(s$rmse, 2)
  )
  df <- data.frame(manual = x, predicted = y)
  
  # Shared axis range
  xy_min <- floor(min(c(x, y), na.rm = TRUE))
  xy_max <- ceiling(max(c(x, y), na.rm = TRUE))
  inset <- 0.01 * (xy_max - xy_min)
  lbl_x <- xy_min + inset
  lbl_y <- xy_max - inset
  
  if (trait == "SL_mm") {
    xy_breaks <- seq(xy_min, xy_max, by = 15)
    p <- ggplot(df, aes(x = manual, y = predicted)) +
      geom_point(colour = "#00F5FF", alpha = 0.5, size = 2) +
      geom_smooth(method = "lm", se = FALSE,
                  colour = "#CD0000", linewidth = 0.7, linetype = "dashed") +
      annotate("label",
               x = lbl_x,
               y = lbl_y,
               label = ann,
               hjust = 0, vjust = 1,
               size  = 3, lineheight = 1.3,
               fill  = "#FFFFF0",
               label.size = 0.5,
               fontface = "bold") +
      scale_x_continuous(breaks = xy_breaks, limits = c(xy_min, xy_max)) +
      scale_y_continuous(breaks = xy_breaks, limits = c(xy_min, xy_max)) +
      coord_cartesian(xlim = c(xy_min, xy_max), ylim = c(xy_min, xy_max)) +
      labs(title = lbl,
           x = axis_labels[[trait]],
           y = y_axis_labels[[trait]]) +
      theme_bw(base_size = 10) +
      theme(
        plot.title       = element_text(face = "bold", size = 11, hjust = 0),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()
      )
    } else {

    # Spikelet traits
    xy_breaks <- seq(xy_min, xy_max, by = 2)
    
    p <- ggplot(df, aes(x = manual, y = predicted)) +
      geom_point(aes(fill = manual),
                 shape = 21, size = 2.0, colour = "#4169E1",
                 stroke = 0.4, alpha = 0.6) +
      scale_fill_gradient(low = "white", high = "#4169E1",
                          name = NULL,
                          breaks = pretty(df$manual, n = 4)) +
      geom_smooth(method = "lm", se = FALSE,
                  colour = "#CD0000", linewidth = 0.7, linetype = "dashed") +
      annotate("label",
               x = lbl_x,
               y = lbl_y,
               label = ann,
               hjust = 0, vjust = 1,
               size  = 3, lineheight = 1.3,
               fill  = "#FFFFF0",
               label.size = 0.5,
               fontface = "bold") +
      scale_x_continuous(breaks = xy_breaks, limits = c(xy_min, xy_max)) +
      scale_y_continuous(breaks = xy_breaks, limits = c(xy_min, xy_max)) +
      coord_cartesian(xlim = c(xy_min, xy_max), ylim = c(xy_min, xy_max)) +
      labs(title = lbl,
           x = axis_labels[[trait]],
           y = y_axis_labels[[trait]]) +
      theme_bw(base_size = 10) +
      theme(
        plot.title           = element_text(face = "bold", size = 11, hjust = 0),
        panel.grid.major     = element_blank(),
        panel.grid.minor     = element_blank(),
        legend.position      = c(0.95, 0.05),
        legend.justification = c(1, 0),
        legend.background    = element_rect(fill = alpha("white", 0.85),
                                            colour = "white", linewidth = 0.3),
        legend.key.height    = unit(0.5, "cm"),
        legend.key.width     = unit(0.3, "cm"),
        legend.text          = element_text(size = 7),
        legend.title         = element_text(size = 8)
      )
  }
  
  plot_list[[trait]] <- p
  ggsave(paste0(trait, "_correlation.jpeg"), plot = p,
         width = 4, height = 4, dpi = 600, device = "png")
}

#Combined figure
combined <- plot_list[[1]] + plot_list[[2]] + plot_list[[3]] + plot_list[[4]] +
  plot_layout(ncol = 2, nrow = 2)

ggsave("all_correlations_combined.jpeg", plot = combined,
       width = 7, height = 7, dpi = 600, device = "jpeg")

#*******************************************************************************
#***************************Figure 2********************************************
library(readxl)
library(dplyr)
library(ggplot2)
library(reshape2)

#validate traits
missing <- setdiff(selected_traits, names(df))
if (length(missing) > 0) {
  cat("  WARNING: Traits not found:", paste(missing, collapse = ", "), "\n")
  selected_traits <- intersect(selected_traits, names(df))
}

# Remove zero-variance columns
mat <- df[, selected_traits, drop = FALSE]
mat <- mat[, sapply(mat, function(x) sd(x, na.rm = TRUE) > 0)]
selected_traits <- names(mat)

cat("  Traits for analysis:", length(selected_traits), "\n")

cat_spike <- c("SL","SA","SP","SR","FZL","SW_api","SW_cen","SW_base",
               "FSk_api","FSk_cen","FSk_base","TFiSk","TSkS","TFSk")   # TSkS added
cat_morph <- c("FSk_A","FSk_L","FSk_P","FSk_R","FSk_W","FSk_gap","FSk_angle")
cat_yield <- c("GCS","GYS","TKW")

trait_category <- dplyr::case_when(
  selected_traits %in% cat_spike ~ "Spike architecture",
  selected_traits %in% cat_morph ~ "Spikelet morphology",
  selected_traits %in% cat_yield ~ "Yield components",
  TRUE ~ "Other")
cat_palette <- c(
  "Spike architecture"  = "#008A02",
  "Spikelet morphology" = "#D33F6A",
  "Yield components"    = "#C26F00"
)
names(trait_category) <- selected_traits

# clean half-square brackets
make_bracket <- function(lo, hi, spine, cap, orient) {
  if (orient == "v") { 
    data.frame(x = c(spine - cap, spine, spine, spine - cap),
               y = c(lo,          lo,    hi,    hi))
  } else {  
    data.frame(x = c(lo,          lo,    hi,    hi),
               y = c(spine + cap, spine, spine, spine + cap))
  }
}

build_brackets <- function(present, orient, spine, cap, lab_gap) {
  pos <- seq_along(present); cats <- trait_category[present]
  paths <- list(); labs <- list(); k <- 0
  for (cc in unique(cats)) {
    idx <- which(cats == cc); if (!length(idx)) next
    k <- k + 1
    lo <- min(pos[idx]) - 0.45; hi <- max(pos[idx]) + 0.45; mid <- (lo + hi)/2 
    paths[[k]] <- cbind(make_bracket(lo, hi, spine, cap, orient), grp = paste(orient, k))
    labs[[k]] <- if (orient == "v") data.frame(x = spine + lab_gap, y = mid, label = cc)
    else                data.frame(x = mid, y = spine - lab_gap, label = cc)
  }
  list(path = do.call(rbind, paths), text = do.call(rbind, labs))
}

#compute correlations with p-value
n_traits   <- length(selected_traits)
cor_mat    <- cor(mat, use = "pairwise.complete.obs", method = "pearson")
# P-value
p_mat <- matrix(NA, n_traits, n_traits)
rownames(p_mat) <- colnames(p_mat) <- selected_traits

for (i in 1:(n_traits - 1)) {
  for (j in (i + 1):n_traits) {
    test <- cor.test(mat[[i]], mat[[j]], method = "pearson")
    p_mat[i, j] <- test$p.value
    p_mat[j, i] <- test$p.value
  }
}
diag(p_mat) <- 0
sig_mat <- matrix("", n_traits, n_traits)
rownames(sig_mat) <- colnames(sig_mat) <- selected_traits

for (i in 1:n_traits) {
  for (j in 1:n_traits) {
    if (i != j && !is.na(p_mat[i, j])) {
      p <- p_mat[i, j]
      sig_mat[i, j] <- ifelse(p < 0.001, "***",
                              ifelse(p < 0.01, "**",
                                     ifelse(p < 0.05, "*", "ns")))
    }
  }
}

#Heat_map figure
melted   <- melt(cor_mat)
names(melted) <- c("Var1", "Var2", "value")

melted_p <- melt(p_mat)
melted$pval <- melted_p$value
melted$sig  <- ifelse(melted$pval < 0.001, "***",
                      ifelse(melted$pval < 0.01, "**",
                             ifelse(melted$pval < 0.05, "*", "")))

melted$show <- TRUE
for (k in 1:nrow(melted)) {
  i <- match(melted$Var1[k], selected_traits)
  j <- match(melted$Var2[k], selected_traits)
  if (!is.na(i) && !is.na(j) && i > j) melted$show[k] <- FALSE
}
melted_upper <- melted %>% filter(show, Var1 != Var2)

x_present <- selected_traits[selected_traits %in% as.character(melted_upper$Var2)]
y_present <- selected_traits[selected_traits %in% as.character(melted_upper$Var1)]
melted_upper$xi <- match(as.character(melted_upper$Var2), x_present)
melted_upper$yi <- match(as.character(melted_upper$Var1), y_present)

y_spine <- 26; y_cap <- 0.3; y_lab <- 1.6
x_spine <- -2.2; x_cap <- 0.5; x_lab <- 1.1
yb <- build_brackets(y_present, "v", y_spine, y_cap, y_lab)
xb <- build_brackets(x_present, "h", x_spine, x_cap, x_lab)
p_heatmap <- ggplot(melted_upper, aes(x = xi, y = yi, fill = value)) +
  geom_tile(color = "white", linewidth = 0.3) +
  geom_text(aes(label = paste0(formatC(round(value, 2), format = "f", digits = 2), "\n", sig)),
            size = 1.8, fontface = "bold", color = "black") +
  geom_path(data = yb$path, aes(x, y, group = grp), inherit.aes = FALSE,
            linewidth = 0.5, linejoin = "mitre") +
  geom_text(data = yb$text, aes(x, y, label = gsub(" ", "\n", label)), inherit.aes = FALSE,
            fontface = "bold", size = 3.2, vjust = 0, lineheigth = 1.0, color = cat_palette) +
  scale_fill_gradient2(low = "#CD0000", mid = "grey95", high = "#4169E1",
                       midpoint = 0, limits = c(-1, 1), breaks = c(-1, 0, 1), name = "Pearson") +
  guides(fill = guide_colorbar(title.position = "top", title.hjust = 0.5)) +
  scale_x_continuous(breaks = seq_along(x_present), labels = x_present,
                     expand = expansion(0)) +
  scale_y_continuous(breaks = seq_along(y_present), labels = y_present,
                     position = "right", expand = expansion(0)) +
  labs(x = NULL, y = NULL) +
  theme_pub +
  theme(
    axis.text.x       = element_text(angle = 45, hjust = 1, size = 8, color = "black"),
    axis.text.y.right = element_text(size = 8, hjust = 0, color = "black",
                                     margin = margin(l = 1, r = 0)),
    legend.position   = c(0.25, 0.90),     
    legend.direction  = "horizontal",
    legend.key.width  = unit(0.5, "cm"),
    legend.key.height = unit(0.2, "cm"),
    legend.background = element_rect(fill = "white", color = NA),
    panel.border      = element_rect(color = "black", fill = NA, linewidth = 0.7),
    plot.margin = margin(t = 5, r = 45, b = 5, l = 5) 
  ) +
  coord_fixed(xlim = c(0.5, length(x_present) + 0.5),
              ylim = c(0.5, length(y_present) + 0.5),
              expand = FALSE, clip = "off")

ggsave(file.path(output_dir, "correlation_heatmap.png"), p_heatmap,
       width = 8, height = 6.5, dpi = 600)

#*******************************************************************************
#*********Figure 3A, 3B, 3C, 3D, 3E, 3F, 3G, 3H, 3I, 3J*************************
library(readxl)
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggrepel)
library(ggforce)
library(dendextend)
library(patchwork)
library(agricolae)
library(fmsb)
library(ggplotify)

N_CLUSTERS <- 4

cluster_traits <- c(
  "SL", "SA", "SR",   "SW_base", "SW_api","FZL",
  "SW_cen","FSk_A", "FSk_W",
  "TSkS",  "TFSk" ,"TFiSk",
  "FSk_P","FSk_L",
  "FSk_angle",  "FSk_api", "FSk_cen" ,"FSk_base" , "FSk_gap", "FSk_R","SP"
)
validation_traits <- c("GCS", "GYS", "TKW")
all_traits        <- c(cluster_traits, validation_traits)

era_col     <- "breeding_era"
ecozone_col <- "ecozone"

cluster_colors <- c("1" = "#A86C34", "2" = "#008A02",
                    "3" = "#2E5A87", "4" = "#CD0000")
cluster_shapes <- c("1" = 16, "2" = 17, "3" = 15, "4" = 18)
era_colors <- c(
  "pre-1950"  = "#CC5260",
  "1951-1980" = "#4AB5C4",
  "1981-2000" = "#DE77AE",
  "post-2000" = "#00A087"
)
ecozone_colors <- c(
  "Founder"        = "#E28B55",
  "Eastern"        = "#BFA554",
  "Western"        = "#9D7660"
)

sig_star <- function(p) {
  if (is.na(p))    return("")
  if (p < 0.001)   return("***")
  if (p < 0.01)    return("**")
  if (p < 0.05)    return("*")
  return("ns")
}

# Run one-way ANOVA + Tukey
run_anova_tukey <- function(data, trait, group = "cluster") {
  fit <- aov(as.formula(paste0("`", trait, "` ~ ", group)), data = data)
  s   <- summary(fit)[[1]]
  p   <- s[group, "Pr(>F)"]
  list(
    f     = s[group, "F value"],
    p     = p,
    sig   = sig_star(p),
    tukey = HSD.test(fit, group, group = TRUE)$groups,
    fit   = fit
  )
}

theme_pub <- theme_minimal(base_size = 12) +
  theme(
    text         = element_text(face = "bold", color = "black"),
    plot.title   = element_text(face = "bold", size = 14, hjust = 0),
    axis.title   = element_text(face = "bold", size = 14),
    axis.text    = element_text(face = "bold", size = 12, color = "black"),
    legend.title = element_text(face = "bold", size = 11),
    legend.text  = element_text(face = "bold", size = 10),
    strip.text   = element_text(face = "bold", size = 11),
    panel.grid.minor = element_blank()
  )

border_theme <- theme(
  panel.border     = element_rect(color = "black", fill = NA, linewidth = 1.0),
  panel.grid.major = element_blank()
)


#PCA
mat <- cultivar_df %>% select(all_of(cluster_traits))
zero_var <- sapply(mat, function(x) sd(x, na.rm = TRUE) == 0)
if (any(zero_var)) mat <- mat[, !zero_var]
for (col in names(mat)) {
  mat[[col]][is.na(mat[[col]])] <- mean(mat[[col]], na.rm = TRUE)
}

pca       <- prcomp(mat, center = TRUE, scale. = TRUE)
prop_var  <- pca$sdev^2 / sum(pca$sdev^2)
cum_var   <- cumsum(prop_var)
n_pc_95   <- which(cum_var >= 0.95)[1]
pc_scores <- pca$x[, 1:n_pc_95, drop = FALSE]

#Hierarchial clustering appraoch

hc       <- hclust(dist(pc_scores, method = "euclidean"), method = "ward.D2")
clusters <- cutree(hc, k = N_CLUSTERS)
cultivar_df$cluster <- as.factor(clusters)

for (cl in sort(unique(clusters))) {
  members <- cultivar_df$variety_name[cultivar_df$cluster == cl]
  cat(sprintf("    Cluster %d (%d): %s\n", cl, length(members),
              paste(members, collapse = ", ")))
}

#Biplots
biplot_df <- data.frame(
  PC1 = pca$x[, 1],
  PC2 = pca$x[, 2],
  variety = cultivar_df$variety_name,
  cluster = cultivar_df$cluster
)
if (has_era)     biplot_df$era     <- cultivar_df[[era_col]]
if (has_ecozone) biplot_df$ecozone <- cultivar_df[[ecozone_col]]
arrow_scale <- min(max(abs(pca$x[, 1])), max(abs(pca$x[, 2]))) * 2.80
arrow_df <- data.frame(
  Trait = rownames(pca$rotation),
  PC1   = pca$rotation[, 1] * arrow_scale,
  PC2   = pca$rotation[, 2] * arrow_scale
)
x_lab <- paste0("PC1 (", round(prop_var[1] * 100, 1), "%)")
y_lab <- paste0("PC2 (", round(prop_var[2] * 100, 1), "%)")

# Hull biplot
p_biplot_hull <- ggplot() +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray70", linewidth = 0.4) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray70", linewidth = 0.4) +
  geom_mark_hull(data = biplot_df,
                 aes(x = PC1, y = PC2, fill = cluster, color = cluster),
                 alpha = 0.12, linewidth = 0.5,
                 expand = unit(3.5, "mm"), radius = unit(3.5, "mm")) +
  geom_point(data = biplot_df,
             aes(x = PC1, y = PC2, color = cluster, shape = cluster), size = 3.5) +
  geom_text_repel(data = biplot_df,
                  aes(x = PC1, y = PC2, label = variety, color = cluster),
                  size = 3.5,
                  max.overlaps = 25, show.legend = FALSE) +
  scale_color_manual(values = cluster_colors, name = "Cluster") +
  scale_fill_manual(values  = cluster_colors, name = "Cluster") +
  scale_shape_manual(values = cluster_shapes, name = "Cluster") +
  labs(x = x_lab, y = y_lab) +
  theme_pub + border_theme +
  theme(legend.position      = c(0.02, 0.02),    # inside panel, bottom-left
        legend.justification = c(0, 0),          # anchor legend's bottom-left corner
        legend.direction     = "horizontal",
        legend.background    = element_rect(fill = alpha("white", 0.85),
                                            color = "black",
                                            linewidth = 0.3),
        legend.title         = element_text(face = "bold", size = 9),
        legend.text          = element_text(face = "bold", size = 8),
        legend.key.size      = unit(0.4, "cm"),
        legend.margin        = margin(3, 6, 3, 6))+
  guides(fill  = guide_legend(override.aes = list(alpha = 0.3)),
         shape = guide_legend(override.aes = list(size = 4))) +
  coord_cartesian(xlim = c(-10, 5), ylim = c(-6, 6))

ggsave(file.path(output_dir, "pca_biplot_clusters_hull.png"),
       p_biplot_hull, width = 10, height = 8, dpi = 600)

#External validation of Breeding era and Ecozone composition
make_composition_plot <- function(data, fill_col, palette, title_label) {
  tab <- table(data$cluster, data[[fill_col]])
  set.seed(123)
  chi <- chisq.test(tab, simulate.p.value = TRUE, B = 10000)
  
  comp <- data %>%
    group_by(cluster, .data[[fill_col]]) %>%
    summarise(n = n(), .groups = "drop") %>%
    group_by(cluster) %>%
    mutate(pct = n / sum(n) * 100)
  
  legend_title <- title_label
  caption_text <- paste0("Chi-sq = ", round(chi$statistic, 2),
                         ", p = ", ifelse(chi$p.value < 0.001, "< 0.001",
                                          formatC(chi$p.value, format = "f", digits = 3)),
                         " ", sig_star(chi$p.value))
  
  list(
    plot = ggplot(comp, aes(x = cluster, y = pct, fill = .data[[fill_col]])) +
      geom_bar(stat = "identity", position = "stack", width = 0.5) +
      geom_text(aes(label = ifelse(pct > 5, n, "")),
                position = position_stack(vjust = 0.5),
                size = 4, fontface = "bold") +
      scale_x_discrete(expand = expansion(add = c(0.3, 0.3))) +
      scale_fill_manual(values = palette, name = legend_title) +
      labs(x = NULL, y = NULL, caption = caption_text) +
      theme_pub + border_theme +
      theme(legend.position = "bottom",
            legend.direction = "horizontal",
            legend.title = element_text(face = "bold", size = 9, hjust = 0.5),
            legend.text  = element_text(face = "bold", size = 9),
            legend.box.margin = margin(t = -5),
            plot.caption = element_text(face = "bold", size = 9,
                                        hjust = 0.5, margin = margin(t = 4))),
    chi   = chi,
    table = tab
  )
}

p_era <- p_eco <- NULL
if (has_era) {
  out <- make_composition_plot(cultivar_df, era_col, era_colors, "Breeding Era")
  p_era <- out$plot
  write.csv(as.data.frame.matrix(out$table),
            file.path(output_dir, "contingency_era.csv"))
  cat(sprintf("  Cluster x Era: chi-sq=%.2f, p=%.4f %s\n",
              out$chi$statistic, out$chi$p.value, sig_star(out$chi$p.value)))
  ggsave(file.path(output_dir, "era_composition.png"), p_era,
         width = 5.5, height = 6, dpi = 600)
}

if (has_ecozone) {
  out <- make_composition_plot(cultivar_df, ecozone_col, ecozone_colors, "Ecozone")
  p_eco <- out$plot
  write.csv(as.data.frame.matrix(out$table),
            file.path(output_dir, "contingency_ecozone.csv"))
  cat(sprintf("  Cluster x Ecozone: chi-sq=%.2f, p=%.4f %s\n",
              out$chi$statistic, out$chi$p.value, sig_star(out$chi$p.value)))
  ggsave(file.path(output_dir, "ecozone_composition.png"), p_eco,
         width = 7, height = 5, dpi = 600)
}

#Radar plots

cluster_means <- cultivar_df %>%
  group_by(cluster) %>%
  summarise(across(all_of(cluster_traits), ~ mean(.x, na.rm = TRUE)),
            .groups = "drop")
cluster_means_norm <- cluster_means
for (trait in cluster_traits) {
  v  <- cluster_means[[trait]]
  mn <- min(v, na.rm = TRUE); mx <- max(v, na.rm = TRUE)
  cluster_means_norm[[trait]] <- if (mx > mn) (v - mn) / (mx - mn) else 0.5
}
make_radar <- function(cl) {
  cl_row <- cluster_means_norm %>%
    filter(cluster == cl) %>%
    select(all_of(cluster_traits))
  radar_df <- rbind(rep(1, length(cluster_traits)),
                    rep(0, length(cluster_traits)),
                    as.numeric(cl_row))
  colnames(radar_df) <- cluster_traits
  radar_df <- as.data.frame(radar_df)
  
  col_line <- cluster_colors[as.character(cl)]
  col_fill <- adjustcolor(col_line, alpha.f = 0.30)
  ggplotify::as.ggplot(function() {
    par(mar = c(2, 2, 2, 2), font = 2)
    radarchart(radar_df, axistype = 0, seg = 3,
               pcol = col_line, pfcol = col_fill, plwd = 2, plty = 1, pty = 16,
               cglcol = "gray50", cglty = 1, cglwd = 0.5, axislabcol = "gray50",
               vlcex = 0.525, vlabcol = "black", calcex = 0.5)
  }) +theme(aspect.ratio = 1,  plot.margin = margin(2, 4, 2, 4))
}

radar_plots <- setNames(
  lapply(sort(unique(as.character(cluster_means$cluster))), make_radar),
  sort(unique(as.character(cluster_means$cluster)))
)
for (cl in names(radar_plots)) {
  ggsave(file.path(output_dir, paste0("radar_cluster_", cl, ".png")),
         radar_plots[[cl]], width = 6, height = 6, dpi = 600, bg = "white")
}
#Yield validation (Violin plots)

make_violin <- function(trait_name) {
  cfg <- validation_config %>% filter(trait == trait_name)
  if (nrow(cfg) == 0) cfg <- tibble(trait = trait_name, label = trait_name,
                                    ymin = NA, ymax = NA)
  
  res <- run_anova_tukey(cultivar_df, trait_name, "cluster")
  annot_text <- paste0("F = ", formatC(res$f, format = "f", digits = 2),
                       ", p = ", ifelse(res$p < 0.001, "< 0.001",
                                        formatC(res$p, format = "f", digits = 3)),
                       " ", res$sig)
  tukey_df <- data.frame(
    cluster     = as.factor(rownames(res$tukey)),
    tukey_group = res$tukey$groups
  )
  y_range <- diff(range(cultivar_df[[trait_name]], na.rm = TRUE))
  per_cluster_max <- cultivar_df %>%
    group_by(cluster) %>%
    summarise(y_top = max(.data[[trait_name]], na.rm = TRUE), .groups = "drop")
  tukey_df <- tukey_df %>%
    left_join(per_cluster_max, by = "cluster") %>%
    mutate(y = y_top + y_range * 0.50)
  
  p <- ggplot(cultivar_df,
              aes(x = cluster, y = .data[[trait_name]], fill = cluster)) +
    geom_violin(alpha = 0.4, linewidth = 0.5, trim = FALSE) +
    geom_boxplot(width = 0.15, alpha = 0.8, outlier.shape = NA, linewidth = 0.4) +
    geom_jitter(width = 0.1, size = 1.5, alpha = 0.6) +
    geom_text(data = tukey_df,
              aes(x = cluster, y = y, label = tukey_group),
              inherit.aes = FALSE, size = 5, fontface = "bold") +
    annotate("text", x = Inf, y = Inf, label = annot_text,
             hjust = 1.1, vjust = 1.8, size = 3.2,
             fontface = "italic") +
    scale_fill_manual(values = cluster_colors) +
    scale_y_continuous(expand = expansion(mult = c(0.05, 0.25))) +
    labs(x = cfg$label, y = NULL) +
    theme_pub + border_theme +
    theme(legend.position = "none")
  
  if (!is.na(cfg$ymin) && !is.na(cfg$ymax)) {
    p <- p + coord_cartesian(ylim = c(cfg$ymin, cfg$ymax))
  }
  p
}

violin_plots <- setNames(
  lapply(validation_traits, make_violin),
  validation_traits
)


for (vn in names(violin_plots)) {
  ggsave(file.path(output_dir, paste0("violin_", vn, ".png")),
         violin_plots[[vn]], width = 5, height = 5, dpi = 600)
}
if (length(violin_plots) >= 2) {
  strip <- wrap_plots(violin_plots, ncol = length(violin_plots))
  ggsave(file.path(output_dir, "yield_validation_violins.png"), strip,
         width = 5 * length(violin_plots), height = 5, dpi = 600)
}
#Boxplots validated using Tukey HSD for selected traits in every cluster [Figure S2]
tukey_results <- data.frame()
for (trait in all_traits) {
  vals <- cultivar_df[[trait]]
  if (all(is.na(vals)) || sd(vals, na.rm = TRUE) == 0) next
  res <- tryCatch(run_anova_tukey(cultivar_df, trait, "cluster"),
                  error = function(e) NULL)
  if (is.null(res)) { cat(sprintf("  %-15s ERROR\n", trait)); next }
  
  groups <- res$tukey
  groups$cluster <- rownames(groups)
  groups$trait   <- trait
  groups$f_value <- round(res$f, 3)
  groups$p_value <- signif(res$p, 4)
  groups$sig     <- res$sig
  names(groups)[names(groups) == "groups"] <- "tukey_group"
  names(groups)[1] <- "mean"
  tukey_results <- rbind(tukey_results, groups)
  cat(sprintf("  %-15s F=%6.2f %3s  %s\n", trait, res$f, res$sig,
              paste(paste0(groups$cluster, "=", groups$tukey_group),
                    collapse = " ")))
}
write.csv(tukey_results, file.path(output_dir, "tukey_hsd_results.csv"),
          row.names = FALSE)

make_boxplot <- function(trait_name) {
  trait_tukey <- tukey_results %>% filter(trait == !!trait_name)
  p <- ggplot(cultivar_df,
              aes(x = cluster, y = .data[[trait_name]], fill = cluster)) +
    geom_boxplot(outlier.shape = 21, outlier.size = 1.5, alpha = 0.7, linewidth = 0.5) +
    geom_jitter(width = 0.15, size = 0.8, alpha = 0.4) +
    scale_fill_manual(values = cluster_colors) +
    labs(title = trait_name, x = NULL, y = NULL) +
    theme_pub + border_theme +
    theme(plot.title = element_text(face = "bold", size = 10, hjust = 0.5),
          legend.position = "none")
  if (nrow(trait_tukey) > 0) {
    y_max   <- max(cultivar_df[[trait_name]], na.rm = TRUE)
    y_range <- diff(range(cultivar_df[[trait_name]], na.rm = TRUE))
    label_df <- trait_tukey %>%
      select(cluster, tukey_group) %>%
      mutate(y = y_max + y_range * 0.08)
    p <- p +
      geom_text(data = label_df,
                aes(x = cluster, y = y, label = tukey_group),
                inherit.aes = FALSE, size = 3.5, fontface = "bold") +
      scale_y_continuous(expand = expansion(mult = c(0.05, 0.15)))
  }
  p
}

box_plots <- lapply(all_traits, function(tr) {
  if (tr %in% names(cultivar_df) && any(!is.na(cultivar_df[[tr]]))) make_boxplot(tr) else NULL
})
box_plots <- box_plots[!sapply(box_plots, is.null)]

if (length(box_plots) > 0) {
  n_rows <- ceiling(length(box_plots) / 5)
  ggsave(file.path(output_dir, "boxplots_all.png"),
         wrap_plots(box_plots, ncol = 5),
         width = 12, height = n_rows * 3, dpi = 600)
}

# Summary tables
summary_table <- cultivar_df %>%
  group_by(cluster) %>%
  summarise(n = n(),
            across(all_of(all_traits),
                   list(mean = ~ round(mean(.x, na.rm = TRUE), 2),
                        sd   = ~ round(sd(.x, na.rm = TRUE), 2)),
                   .names = "{.col}__{.fn}"),
            .groups = "drop")
write.csv(summary_table, file.path(output_dir, "cluster_summary.csv"),
          row.names = FALSE)

formatted <- data.frame(Trait = character(), stringsAsFactors = FALSE)
for (trait in all_traits) {
  row <- data.frame(Trait = trait)
  for (cl in sort(unique(na.omit(cultivar_df$cluster)))) {
    m  <- summary_table[[paste0(trait, "__mean")]][summary_table$cluster == cl]
    s  <- summary_table[[paste0(trait, "__sd")]][summary_table$cluster == cl]
    tg <- tukey_results %>%
      filter(trait == !!trait, cluster == as.character(cl)) %>%
      pull(tukey_group)
    row[[paste0("Cl_", cl)]] <- paste0(m, " +/- ", s, " ",
                                       ifelse(length(tg) > 0, tg[1], ""))
  }
  formatted <- rbind(formatted, row)
}
write.csv(formatted, file.path(output_dir, "cluster_summary_formatted.csv"),
          row.names = FALSE)

#Combined figure

p_pca_combined <- p_biplot_hull
clust_keys  <- sort(names(radar_plots))[1:min(4, length(radar_plots))]
radar_grid  <- (radar_plots[[clust_keys[1]]] | radar_plots[[clust_keys[2]]]) /
  (radar_plots[[clust_keys[3]]] | radar_plots[[clust_keys[4]]])

top_row <- p_pca_combined | radar_grid
top_row <- top_row + plot_layout(widths = c(1.2, 1))

mid_pieces <- list()
if (!is.null(p_era)) mid_pieces[[length(mid_pieces) + 1]] <- p_era
if (!is.null(p_eco)) mid_pieces[[length(mid_pieces) + 1]] <- p_eco
mid_row <- if (length(mid_pieces) == 2) {
  mid_pieces[[1]] | mid_pieces[[2]]
} else if (length(mid_pieces) == 1) {
  mid_pieces[[1]]
} else {
  NULL
}
bot_row <- wrap_plots(violin_plots, ncol = length(violin_plots))

fig <- top_row
if (!is.null(mid_row)) fig <- fig / mid_row
fig <- fig / bot_row

fig <- fig +
  plot_annotation(tag_levels = "A") &
  theme(plot.tag = element_text(face = "bold", size = 16))
n_rows_fig <- 1 + as.integer(!is.null(mid_row)) + 1
heights <- if (n_rows_fig == 3) c(2.2, 1.4, 1.2) else c(2.2, 1.2)
fig <- fig + plot_layout(heights = heights)

ggsave(file.path(output_dir, "combined_figure.png"),
       fig, width = 12, height = 14, dpi = 600, bg = "white")

#*******************************************************************************
#***************************Figure 4********************************************
library(readxl)
library(dplyr)
library(ggplot2)
trait_cols <- c(
  "SL", "SA", "SP", "SR", "FZL",
  "SW_api", "SW_cen", "SW_base",
  "FSk_api", "FSk_cen", "FSk_base",
  "TFiSk", "TSkS","TFSk",
  "FSk_angle", "FSk_L", "FSk_W", "FSk_A", "FSk_P", "FSk_R", "FSk_gap",
  "GCS", "GYS", "TKW"
)

#Fixed decade bins
decade_bins <- list(
  "D1840" = c(1841, 1900), 
  "D1900" = c(1901, 1930),
  "D1930" = c(1931, 1960),
  "D1960" = c(1961, 1970),
  "D1970" = c(1971, 1980),
  "D1980" = c(1981, 1990),
  "D1990" = c(1991, 2000),
  "D2000" = c(2001, 2010),
  "D2010" = c(2011, 2020)
)
decade_names <- names(decade_bins)
col_increase <- "#4169E1"
col_decrease <- "#CD0000"
theme_pub <- theme_minimal(base_size = 12) +
  theme(
    text         = element_text(face = "bold", color = "black"),
    axis.title   = element_text(face = "bold", size = 12),
    axis.text    = element_text(face = "bold", size = 10, color = "black"),
    legend.title = element_text(face = "bold", size = 11),
    legend.text  = element_text(face = "bold", size = 10),
    panel.grid   = element_blank()
  )


#Assign decadal bins
assign_decade <- function(year) {
  for (d in names(decade_bins)) {
    rng <- decade_bins[[d]]
    if (year >= rng[1] && year <= rng[2]) return(d)
  }
  return(NA_character_)
}

cultivar_df$decade <- sapply(cultivar_df$yor, assign_decade)
cultivar_df <- cultivar_df %>% filter(!is.na(decade))

decade_counts <- cultivar_df %>%
  group_by(decade) %>%
  summarise(n = n(), years = paste0(min(yor), "-", max(yor)), .groups = "drop") %>%
  mutate(decade = factor(decade, levels = decade_names)) %>%
  arrange(decade)

for (i in 1:nrow(decade_counts)) {
  cat(sprintf("    %s: %d cultivars (%s)\n",
              decade_counts$decade[i], decade_counts$n[i], decade_counts$years[i]))
}

#Decadal means
decade_means <- cultivar_df %>%
  group_by(decade) %>%
  summarise(across(all_of(trait_cols), ~ mean(.x, na.rm = TRUE)),
            .groups = "drop") %>%
  mutate(decade = factor(decade, levels = decade_names)) %>%
  arrange(decade)

#Output file
results <- data.frame()

for (trait in trait_cols) {
  vals <- decade_means[[trait]]
  names(vals) <- as.character(decade_means$decade)
  
  if (all(is.na(vals)) || sd(vals, na.rm = TRUE) == 0) next
  
  # Start row
  row <- data.frame(Trait = trait, stringsAsFactors = FALSE)
  
  # Add each decade column
  for (d in decade_names) {
    if (d %in% names(vals) && !is.na(vals[d])) {
      row[[d]] <- round(as.numeric(vals[d]), 4)
    } else {
      row[[d]] <- NA
    }
  }
  
  # Total change
  valid_vals   <- vals[!is.na(vals)]
  first_val    <- valid_vals[1]
  last_val     <- valid_vals[length(valid_vals)]
  total_change <- ((last_val - first_val) / abs(first_val)) * 100
  
  # Rate per decade
  valid_decades <- decade_names[decade_names %in% names(valid_vals)]
  rate_per_dec  <- total_change / 17.6
  
  direction <- ifelse(total_change >= 0, "Increase", "Decrease")
  
  row[["Total_Change (%)"]]    <- round(total_change, 2)
  row[["Rate_per_decade (%)"]] <- round(rate_per_dec, 2)
  row[["Direction"]]           <- direction
  
  results <- rbind(results, row)
  
  cat(sprintf("  %-15s  %s -> %s:  total = %+.1f%%  rate = %+.1f%%/decade  [%s]\n",
              trait,
              names(valid_vals)[1], names(valid_vals)[length(valid_vals)],
              total_change, rate_per_dec, direction))
}

results <- results %>% arrange(`Rate_per_decade (%)`)
#Output figure
plot_data <- results %>%
  mutate(
    Trait     = factor(Trait, levels = Trait),
    label     = sprintf("%+.1f%%", `Rate_per_decade (%)`),
    rate      = `Rate_per_decade (%)`,
    direction = Direction
  )
p <- ggplot(plot_data, aes(x = rate, y = Trait)) +
  geom_segment(aes(x = 0, xend = rate, y = Trait, yend = Trait,
                   color = direction),
               linewidth = 1.2) +
  geom_point(aes(color = direction), size = 3.5) +
  geom_text(aes(label = label,
                hjust = ifelse(rate >= 0, -0.3, 1.3)),
            size = 3.2, fontface = "bold", color = "black") +
  geom_vline(xintercept = 0, linewidth = 0.6, color = "black") +
  scale_color_manual(values = c("Increase" = col_increase,
                                "Decrease" = col_decrease),
                     name = NULL) +
  scale_x_continuous(expand = expansion(mult = c(0.15, 0.15))) +
  labs(title = NULL,
       x     = "Rate of Change per Decade (%)",
       y     = NULL) +
  theme_pub +
  theme(legend.position = "bottom",
        panel.border    = element_rect(color = "black", fill = NA, linewidth = 1.0))

fig_height <- max(6, nrow(plot_data) * 0.35)

ggsave(file.path(output_dir, "decadal_rate_lollipop.png"), p,
       width = 7, height = fig_height, dpi = 600)

#*******************************************************************************
#***************************Figure 6A, 6B, 6C***********************************
library(readxl)
library(dplyr)
library(lavaan)
library(semPlot)
library(ggplot2)
library(patchwork)
library(png)
library(grid)

#Hypothesized path model
model_spec <- '
  TFiSk ~ SL
  FSk_angle ~ SL
  SA    ~ SL  + FSk_angle + TFiSk
  SW    ~ SL + SA + FSk_angle + TFiSk
  TFSk  ~ SL+ SA + FSk_angle + SW
  FSk_gap ~ SL + TFSk + TFiSk + SW 
  FSk_A   ~ FSk_gap + TFSk + SL + SW + SA
  GCS   ~ SW + TFSk + SA + SL + FSk_A
  GYS   ~ GCS + SW  + FSk_gap

  #Residual Covariances added from MI
  FSk_gap ~~ GCS
  SW ~~ GYS
'
model_vars <- c("SL", "SA", "SW", "TFSk",  "TFiSk",
                "FSk_A", "FSk_gap", "FSk_angle", "GCS", "GYS")

cat_spike <- c("SL", "SA", "SW", "TFSk", "TFiSk")
cat_morph <- c("FSk_A", "FSk_gap", "FSk_angle")
cat_yield <- c("GCS", "GYS")

cat_levels  <- c("Spike architecture", "Spikelet morphology",
                 "Yield components", "Unexplained")
cat_palette <- c(
  "Spike architecture"  = "#8CCD8C",
  "Spikelet morphology" = "#F9A5CD",
  "Yield components"    = "#FDAA65",
  "Unexplained"         = "#E0E0E0"
)

sig_star <- function(p) {
  if (is.na(p))  return("ns")
  if (p < 0.001) return("***")
  if (p < 0.01)  return("**")
  if (p < 0.05)  return("*")
  if (p < 0.1)   return(".")
  "ns"
}

get_cat_label <- function(v) {
  if (v %in% cat_spike) return("Spike architecture")
  if (v %in% cat_morph) return("Spikelet morphology")
  if (v %in% cat_yield) return("Yield components")
  "Other"
}
# Direct / indirect / total effects
effect_decomposition <- function(fit, vars) {
  ss  <- standardizedSolution(fit)
  reg <- ss[ss$op == "~", c("lhs", "rhs", "est.std")]
  B <- matrix(0, length(vars), length(vars), dimnames = list(vars, vars))
  for (i in seq_len(nrow(reg)))
    if (reg$lhs[i] %in% vars && reg$rhs[i] %in% vars)
      B[reg$lhs[i], reg$rhs[i]] <- reg$est.std[i]   # B[outcome, predictor]
  I_mat <- diag(length(vars))
  Tot <- tryCatch(solve(I_mat - B) - I_mat, error = function(e) NULL)
  if (is.null(Tot)) { warning("Effect decomposition failed (non-recursive?)."); return(data.frame()) }
  Ind <- Tot - B
  out <- expand.grid(to = vars, from = vars, stringsAsFactors = FALSE)
  out$direct   <- mapply(function(t, f) B[t, f],   out$to, out$from)
  out$indirect <- mapply(function(t, f) Ind[t, f], out$to, out$from)
  out$total    <- mapply(function(t, f) Tot[t, f], out$to, out$from)
  out <- out[abs(out$total) > 1e-8 | abs(out$direct) > 1e-8, c("from", "to", "direct", "indirect", "total")]
  out[order(-abs(out$total)), ]
}
all_dir_paths <- function(from, to, edges) {
  out <- list()
  dfs <- function(node, acc) {
    nx <- edges[edges$from == node, , drop = FALSE]
    if (nrow(nx) == 0) return(invisible())
    for (i in seq_len(nrow(nx))) {
      lab <- nx$label[i]; nxt <- nx$to[i]
      if (nxt == to) out[[length(out) + 1]] <<- c(acc, lab)
      else dfs(nxt, c(acc, lab))
    }
  }
  dfs(from, character(0)); out
}

build_labelled_model <- function(fit, targets) {
  pe  <- parameterEstimates(fit)
  edg <- pe[pe$op == "~", c("lhs", "rhs")]; names(edg) <- c("to", "from")
  edg$label <- paste0("b_", edg$to, "_", edg$from)
  reg_lines <- vapply(split(edg, edg$to), function(g)
    paste0(g$to[1], " ~ ", paste0(g$label, "*", g$from, collapse = " + ")), character(1))
  rc <- pe[pe$op == "~~" & pe$lhs != pe$rhs, c("lhs", "rhs")]
  rc_lines <- if (nrow(rc) > 0) paste0(rc$lhs, " ~~ ", rc$rhs) else character(0)
  def_lines <- character(0)
  for (t in targets) {
    for (s in setdiff(unique(edg$from), t)) {
      paths <- all_dir_paths(s, t, edg)
      if (length(paths) == 0) next
      prod_strs <- vapply(paths, function(p) paste0("(", paste(p, collapse = "*"), ")"), character(1))
      def_lines <- c(def_lines, sprintf("total_%s_%s := %s", t, s, paste(prod_strs, collapse = " + ")))
      ind_paths <- paths[vapply(paths, length, integer(1)) > 1L]
      if (length(ind_paths) > 0) {
        ind_expr <- paste(vapply(ind_paths, function(p) paste0("(", paste(p, collapse = "*"), ")"), character(1)), collapse = " + ")
        def_lines <- c(def_lines, sprintf("indirect_%s_%s := %s", t, s, ind_expr))
      }
    }
  }
  paste(c(reg_lines, rc_lines, def_lines), collapse = "\n")
}

#Calculate fit
df <- read_excel(input_file, sheet = input_sheet)
names(df) <- gsub("-", "_", names(df))
df_scaled <- df
df_scaled[model_vars] <- lapply(df_scaled[model_vars], function(x) {
  x  <- as.numeric(x)
  sx <- sd(x, na.rm = TRUE)
  if (is.finite(sx) && sx > 0) as.numeric(scale(x)) else x
})

fit <- sem(model_spec, data = df_scaled, estimator = "MLR",
           cluster = if (use_cluster) cluster_var else NULL)


#Fit measures
fm_keys <- c("chisq", "df", "pvalue", "cfi", "tli",
             "rmsea", "srmr")
fm <- fitMeasures(fit, fm_keys)
cat(sprintf("\n  Chi-sq = %.2f  df = %.0f  p = %.4f\n", fm["chisq"], fm["df"], fm["pvalue"]))
cat(sprintf("  CFI = %.3f  TLI = %.3f  RMSEA = %.3f [%.3f, %.3f]  SRMR = %.3f\n",
            fm["cfi"], fm["tli"], fm["rmsea"], fm["srmr"]))
write.csv(data.frame(Measure = names(fm), Value = round(as.numeric(fm), 4)),
          file.path(output_dir, "model_fit.csv"), row.names = FALSE)

#R2 and standardized error variances(1-R2)
r2    <- inspect(fit, "r2")
r2_df <- data.frame(Variable = names(r2), R_squared = round(as.numeric(r2), 4)) %>%
  dplyr::arrange(dplyr::desc(R_squared))
write.csv(r2_df, file.path(output_dir, "r_squared.csv"), row.names = FALSE)
err_std <- data.frame(Variable = names(r2),
                      error_variance_std = round(1 - as.numeric(r2), 4)) %>%
  dplyr::arrange(dplyr::desc(error_variance_std))
write.csv(err_std, file.path(output_dir, "error_variances_standardized.csv"), row.names = FALSE)

#Path coefficients (combined standardized and unstandardized estimates)
unstd <- parameterEstimates(fit, ci = TRUE) %>%
  dplyr::filter(op == "~") %>%
  dplyr::transmute(lhs, rhs, est_unstd = est, se, CR = z, pvalue, ci.lower, ci.upper)
std <- standardizedSolution(fit) %>%
  dplyr::filter(op == "~") %>%
  dplyr::transmute(lhs, rhs, est_std = est.std)
paths <- dplyr::left_join(unstd, std, by = c("lhs", "rhs")) %>%
  dplyr::mutate(path = paste(rhs, "->", lhs),
                sig  = vapply(pvalue, sig_star, character(1))) %>%
  dplyr::arrange(lhs, dplyr::desc(abs(est_std)))
write.csv(paths, file.path(output_dir, "path_coefficients.csv"), row.names = FALSE)

#DIRECT, INDIRECT, and TOTAL EFFECTS
effects_all <- effect_decomposition(fit, model_vars)
write.csv(transform(effects_all,
                    direct = round(direct, 4), indirect = round(indirect, 4), total = round(total, 4)),
          file.path(output_dir, "effects_decomposition.csv"), row.names = FALSE)

yc <- effects_all %>% dplyr::filter(to %in% yield_targets) %>%
  dplyr::mutate(
    pct_indirect = ifelse(abs(total) > 1e-8, round(100 * indirect / total, 1), NA_real_),
    route = dplyr::case_when(
      abs(direct) < 1e-8 & abs(indirect) > 1e-8             ~ "purely indirect",
      abs(indirect) < 1e-8                                  ~ "purely direct",
      sign(direct) != sign(indirect) & abs(indirect) > 1e-8 ~ "mixed / suppression",
      abs(indirect) > abs(direct)                           ~ "mostly indirect",
      TRUE                                                  ~ "mostly direct"),
    direct = round(direct, 4), indirect = round(indirect, 4), total = round(total, 4)) %>%
  dplyr::arrange(to, dplyr::desc(abs(total)))
write.csv(yc, file.path(output_dir, "effects_yield_centric.csv"), row.names = FALSE)

#Modification Indices (MI)
mi_df <- tryCatch({
  modificationIndices(fit, sort. = TRUE) %>%
    dplyr::filter(!is.na(mi)) %>%
    dplyr::mutate(
      type = dplyr::case_when(
        op == "~"  ~ "Suggested path",
        op == "~~" ~ "Suggested residual covariance",
        op == "=~" ~ "Suggested factor loading",
        TRUE       ~ paste0("Other (", op, ")")),
      relation = dplyr::case_when(
        op == "~~" ~ paste(lhs, "<->", rhs),
        op == "=~" ~ paste(lhs, "=~", rhs),
        TRUE       ~ paste(rhs, "->", lhs)),
      strength = dplyr::case_when(mi > 10 ~ "strong", mi > 4 ~ "moderate", TRUE ~ "weak")) %>%
    dplyr::select(type, relation, mi, epc, sepc.all, strength) %>%
    head(n_top_mi)
}, error = function(e) { warning("Modification indices failed: ", conditionMessage(e)); data.frame() })
write.csv(mi_df, file.path(output_dir, "modification_indices.csv"), row.names = FALSE)

resids <- tryCatch(lavResiduals(fit, type = "cor.bentler")$cov.z, error = function(e) NULL)
if (!is.null(resids)) {
  resid_df <- as.data.frame(as.table(resids)) %>%
    dplyr::filter(as.character(Var1) < as.character(Var2)) %>%
    dplyr::rename(Var_1 = Var1, Var_2 = Var2, z = Freq) %>%
    dplyr::mutate(z = round(z, 3), flag = ifelse(abs(z) > 2.58, "LARGE", "")) %>%
    dplyr::arrange(dplyr::desc(abs(z)))
  write.csv(resid_df, file.path(output_dir, "residuals.csv"), row.names = FALSE)
}
#Path diagram
node_colors <- vapply(model_vars, function(v) {
  cl <- get_cat_label(v)
  unname(if (cl %in% names(cat_palette)) cat_palette[cl] else "#FFFFF0")
}, character(1))
names(node_colors) <- model_vars

node_order  <- lavaan::lavNames(fit, "ov")
node_colors <- node_colors[node_order]
stopifnot(identical(sort(names(node_colors)), sort(model_vars)))

draw_diagram <- function() {
  semPlot::semPaths(
    fit,
    whatLabels = "std", what = "std",
    layout = "circle", style = "lisrel",
    residuals = FALSE, intercepts = FALSE, fade = FALSE,
    posCol = "#4169E1", negCol = "#CD0000", negDashed = TRUE,
    nodeLabels = node_order,
    color = list(man = node_colors, lat = character(0)), border.color = "black", border.width = 1.5,
    shapeMan = "rectangle", sizeMan = 7, sizeMan2 = 4,
    label.cex = 1.1, label.font = 1, label.color = "black",
    nCharNodes = 0,
    edge.width = 0.5, edge.label.color = "black", edge.label.position = 0.80, edge.label.cex = 1.0,
    asize = 2.5, curve = 0, curvePivot = TRUE,
    mar = c(2, 2, 2, 2)
  )
  fit_text <- c(
    sprintf("\u03C7\u00B2 = %.2f  (df = %.0f, p = %.3f)", fm["chisq"], fm["df"], fm["pvalue"]),
    sprintf("CFI = %.3f ;  TLI = %.3f", fm["cfi"], fm["tli"]),
    sprintf("RMSEA = %.3f ;  SRMR = %.3f", fm["rmsea"], fm["srmr"])
  )
  legend("bottomright", legend = fit_text,
         bg = "#FFFFF0", box.col = "black", box.lwd = 0.5,
         cex = 0.7, inset = c(0.003, 0.005), y.intersp = 1.0, adj = c(0, 0.5), x.intersp = 0.2)
}

path_png <- file.path(output_dir, "path_diagram.png")
png(path_png, width = 7, height = 6, units = "in", res = 600, bg = "white")
par(mar = c(0, 0, 0, 0)); draw_diagram(); dev.off()

path_img <- png::readPNG(path_png)
p_A <- wrap_elements(full = rasterGrob(path_img, interpolate = TRUE)) +
  labs(tag = "A") +
  theme(plot.tag = element_text(face = "bold", size = 20),
        plot.tag.position = c(0.09, 0.97), plot.margin = margin(2, 2, 2, 2))

#Donut plots decomposed by total effects

total_effects_on <- function(target) {
  e <- effects_all[effects_all$to == target, c("from", "total")]
  setNames(e$total, e$from)
}

compute_category_contrib_total <- function(target) {
  te <- total_effects_on(target); preds <- names(te)
  if (length(preds) == 0) return(NULL)
  raw <- vapply(preds, function(p)
    if (p %in% rownames(cor_mat)) te[[p]] * cor_mat[p, target] else NA_real_, numeric(1))
  data.frame(rhs = preds, raw = as.numeric(raw),
             category = vapply(preds, get_cat_label, character(1)),
             stringsAsFactors = FALSE) %>%
    dplyr::filter(!is.na(raw)) %>%
    dplyr::group_by(category) %>%
    dplyr::summarise(value_signed = sum(raw), .groups = "drop")
}

make_donut <- function(target) {
  contrib <- compute_category_contrib_total(target)
  r2_val  <- as.numeric(r2[target])
  if (is.null(contrib) || is.na(r2_val)) return(NULL)
  contrib$is_neg <- contrib$value_signed < 0
  pos_sum <- sum(contrib$value_signed[contrib$value_signed > 0])
  neg_sum <- sum(abs(contrib$value_signed[contrib$value_signed < 0]))
  total_mag <- pos_sum + neg_sum
  contrib$arc <- dplyr::case_when(
    !contrib$is_neg & total_mag > 0 ~
      (contrib$value_signed / total_mag) * r2_val,
    contrib$is_neg & total_mag > 0  ~
      (abs(contrib$value_signed) / total_mag) * r2_val,
    TRUE ~ 0
  )
  ggplot(outer_df) +
    geom_rect(aes(xmin = 2.5, xmax = 4, ymin = ymin, ymax = ymax, fill = category, color = brdr),
              linewidth = 0.6) +
    scale_color_identity() +
    geom_text(aes(x = 3.25, y = (ymin + ymax) / 2, label = pct_label),
              color = "black", fontface = "bold", size = 3.5) +
    annotate("text", x = 0, y = 0, label = paste0(target, "\nR\u00B2 = ", round(r2_val, 3)),
             size = 4.2, fontface = "bold", color = "black") +
    scale_fill_manual(values = cat_palette, limits = cat_levels, drop = FALSE, name = NULL) +
    coord_polar(theta = "y") + xlim(c(0, 4.5)) +
    theme_void(base_size = 11) +
    theme(legend.position = "none", plot.margin = margin(0, 0, 0, 0))
}

donut_gcs <- make_donut("GCS")
donut_gys <- make_donut("GYS")
ggsave(file.path(output_dir, "donut_GCS.png"), donut_gcs, width = 4, height = 4, dpi = 600, bg = "white")
ggsave(file.path(output_dir, "donut_GYS.png"), donut_gys, width = 4, height = 4, dpi = 600, bg = "white")

#Legend
common_legend <- ggplot() +
  annotate("segment", x = 0.01, xend = 0.05, y = 0.5, yend = 0.5, color = "#4169E1", linewidth = 1.0,
           arrow = arrow(length = unit(0.12, "cm"), type = "closed")) +
  annotate("text", x = 0.06, y = 0.5, label = "Positive", hjust = 0, size = 3.6, fontface = "bold") +
  annotate("segment", x = 0.18, xend = 0.23, y = 0.5, yend = 0.5, color = "#CD0000", linewidth = 1.0,
           linetype = "dashed", arrow = arrow(length = unit(0.12, "cm"), type = "closed")) +
  annotate("text", x = 0.24, y = 0.5, label = "Negative", hjust = 0, size = 3.6, fontface = "bold") +
  annotate("rect", xmin = 0.38, xmax = 0.42, ymin = 0.42, ymax = 0.58, fill = cat_palette["Spike architecture"]) +
  annotate("text", x = 0.435, y = 0.5, label = "Spike architecture", hjust = 0, size = 3.6) +
  annotate("rect", xmin = 0.62, xmax = 0.66, ymin = 0.42, ymax = 0.58, fill = cat_palette["Spikelet morphology"]) +
  annotate("text", x = 0.67, y = 0.5, label = "Spikelet morphology", hjust = 0, size = 3.6) +
  annotate("rect", xmin = 0.88, xmax = 0.92, ymin = 0.42, ymax = 0.58, fill = cat_palette["Yield components"]) +
  annotate("text", x = 0.93, y = 0.5, label = "Yield components", hjust = 0, size = 3.6) +
  annotate("rect", xmin = 1.13, xmax = 1.17, ymin = 0.42, ymax = 0.58,
           fill = cat_palette["Unexplained"], color = "grey60", linewidth = 0.3) +
  annotate("text", x = 1.18, y = 0.5, label = "Unexplained", hjust = 0, size = 3.6) +
  xlim(0, 1.28) + ylim(0, 1) + theme_void() + theme(plot.margin = margin(0, 0, 0, 0))

#Combined figure
p_B <- donut_gcs + labs(tag = "B") +
  theme(plot.tag = element_text(face = "bold", size = 20),
        plot.tag.position = c(0.05, 0.95), plot.margin = margin(2, 2, 2, 2))
p_C <- donut_gys + labs(tag = "C") +
  theme(plot.tag = element_text(face = "bold", size = 20),
        plot.tag.position = c(0.05, 0.95), plot.margin = margin(0, 0, 0, 0))

donut_col <- p_B / p_C                       # GCS on top, GYS below
top_row   <- (p_A | donut_col) + plot_layout(widths = c(2, 1))
combined  <- (top_row / common_legend) + plot_layout(heights = c(1, 0.05)) &
  theme(plot.margin = margin(0, 1, 0, 1))

ggsave(file.path(output_dir, "combined_figure.png"), combined,
       width = 10, height = 7, dpi = 600, bg = "white")

#*******************************************************************************
#*****************Figure S3*****************************************************
library(readxl)
library(ggplot2)
library(patchwork)
library(dplyr)
library(tidyr)

breakpoint_year <- 1970

col_increase <- "#4169E1"  
col_decrease <- "#CD0000" 

decade_bins <- list(
  "D1840" = c(1841, 1900), 
  "D1900" = c(1901, 1930),
  "D1930" = c(1931, 1960),
  "D1960" = c(1961, 1970),
  "D1970" = c(1971, 1980),
  "D1980" = c(1981, 1990),
  "D1990" = c(1991, 2000),
  "D2000" = c(2001, 2010),
  "D2010" = c(2011, 2020)
)
decade_breaks <- sapply(decade_bins, mean)
decade_labels <- names(decade_bins)

selected_traits <- c(
  "SL", "SA", "SP", "SR",
  "SW_api", "SW_cen", "SW_base",
  "FSk_api", "FSk_cen", "FSk_base",
  "TFSk", "TFiSk", "TSkS",
  "FZL",
  "FSk_angle", "FSk_L", "FSk_W", "FSk_A", "FSk_P", "FSk_R", "FSk_gap",
  "GCS", "GYS", "TKW"
)

ncol_combined <- 5

theme_pub <- theme_minimal(base_size = 12) +
  theme(
    text             = element_text(face = "bold", color = "black"),
    plot.title       = element_text(face = "bold", size = 13, hjust = 0.5),
    plot.subtitle    = element_text(size = 9, hjust = 0.5, color = "gray40"),
    axis.title       = element_text(face = "bold", size = 11),
    axis.text        = element_text(face = "bold", size = 10, color = "black"),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border     = element_rect(color = "black", fill = NA, linewidth = 0.8)
  )

sig_star <- function(p) {
  if (is.na(p))    return("")
  if (p < 0.001)   return("***")
  if (p < 0.01)    return("**")
  if (p < 0.05)    return("*")
  return("ns")
}

#Regression fit before 1970 vs after 1970

breakpoint_results <- data.frame()
all_plots <- list()

for (trait in selected_traits) {
  vals  <- cultivar_df[[trait]]
  years <- cultivar_df$yor
  
  if (all(is.na(vals)) || sd(vals, na.rm = TRUE) == 0) {
    cat(sprintf("  %-20s  SKIPPED (no variation)\n", trait))
    next
  }
  
  tryCatch({
    idx_before <- which(years <  breakpoint_year & !is.na(vals))
    idx_after  <- which(years >= breakpoint_year & !is.na(vals))
    
    if (length(idx_before) < 2 || length(idx_after) < 2) {
      cat(sprintf("  %-20s  SKIPPED (need >=2 points each side)\n", trait))
      next
    }
    data_before <- data.frame(year = years[idx_before], value = vals[idx_before])
    data_after  <- data.frame(year = years[idx_after],  value = vals[idx_after])
    data_all    <- data.frame(year = years,             value = vals)
    fit_linear <- lm(value ~ year, data = data_all)
    r2_linear  <- summary(fit_linear)$r.squared
    ss_lin     <- sum(residuals(fit_linear)^2, na.rm = TRUE)
    fit_before <- lm(value ~ year, data = data_before)
    fit_after  <- lm(value ~ year, data = data_after)
    slope_before <- coef(fit_before)[2]
    slope_after  <- coef(fit_after)[2]
    p_before <- summary(fit_before)$coefficients[2, "Pr(>|t|)"]
    p_after  <- summary(fit_after)$coefficients[2, "Pr(>|t|)"]
    sig_before <- sig_star(p_before)
    sig_after  <- sig_star(p_after)
    pred_combined <- rep(NA_real_, length(vals))
    pred_combined[idx_before] <- predict(fit_before)
    pred_combined[idx_after]  <- predict(fit_after)
    ss_seg <- sum((vals - pred_combined)^2, na.rm = TRUE)
    ss_tot <- sum((vals - mean(vals, na.rm = TRUE))^2, na.rm = TRUE)
    r2_pool <- 1 - ss_seg / ss_tot
    n_total <- length(idx_before) + length(idx_after)
    df1 <- 2
    df2 <- n_total - 4
    if (df2 > 0 && ss_seg > 0) {
      F_stat  <- ((ss_lin - ss_seg) / df1) / (ss_seg / df2)
      p_break <- pf(F_stat, df1, df2, lower.tail = FALSE)
    } else {
      F_stat <- NA; p_break <- NA
    }
    sig_break <- sig_star(p_break)
    
    cat(sprintf("  %-20s  %+12.5f%-2s  %+12.5f%-2s  %8.3f  %8.3f  %12s\n",
                trait,
                slope_before, sig_before,
                slope_after,  sig_after,
                r2_linear, r2_pool,
                ifelse(is.na(p_break), "NA",
                       paste0(signif(p_break, 3), sig_break))))
    
    breakpoint_results <- rbind(breakpoint_results, data.frame(
      Trait            = trait,
      Breakpoint_Year  = breakpoint_year,
      N_before         = length(idx_before),
      N_after          = length(idx_after),
      Slope_before     = round(slope_before, 6),
      Slope_after      = round(slope_after,  6),
      P_before         = signif(p_before, 4),
      P_after          = signif(p_after,  4),
      Sig_before       = sig_before,
      Sig_after        = sig_after,
      Direction_before = ifelse(slope_before >= 0, "Increase", "Decrease"),
      Direction_after  = ifelse(slope_after  >= 0, "Increase", "Decrease"),
      R2_linear        = round(r2_linear, 3),
      R2_pooled        = round(r2_pool,   3),
      R2_improvement   = round(r2_pool - r2_linear, 3),
      F_stat_break     = ifelse(is.na(F_stat),  NA, round(F_stat, 3)),
      P_break          = ifelse(is.na(p_break), NA, signif(p_break, 4)),
      Sig_break        = sig_break,
      stringsAsFactors = FALSE
    ))
    yr_before <- seq(min(years[idx_before]), breakpoint_year, length.out = 100)
    yr_after  <- seq(breakpoint_year, max(years[idx_after]),  length.out = 100)
    
    pred_before_df <- data.frame(
      year  = yr_before,
      value = predict(fit_before, newdata = data.frame(year = yr_before))
    )
    pred_after_df <- data.frame(
      year  = yr_after,
      value = predict(fit_after,  newdata = data.frame(year = yr_after))
    )
    
    color_before <- ifelse(slope_before >= 0, col_increase, col_decrease)
    color_after  <- ifelse(slope_after  >= 0, col_increase, col_decrease)
    stats_label <- paste0(
      "BP: ", breakpoint_year, "\n",
      "Before: ", sprintf("%+.3f", slope_before), " ", sig_before, "\n",
      "After:  ", sprintf("%+.3f", slope_after),  " ", sig_after,  "\n",
      "R\u00b2 = ", sprintf("%.3f", r2_pool),
      "  (F-test ", sig_break, ")"
    )
    p <- ggplot() +
      geom_line(data = pred_before_df,
                aes(x = year, y = value),
                color = color_before, linewidth = 1.0, linetype = "solid") +
      geom_line(data = pred_after_df,
                aes(x = year, y = value),
                color = color_after, linewidth = 1.0, linetype = "solid") +
      geom_point(data = cultivar_df,
                 aes(x = yor, y = .data[[trait]]),
                 size = 1.8, alpha = 0.75, color = "#997227") +
      geom_vline(xintercept = breakpoint_year,
                 linetype = "dashed", color = "gray50", linewidth = 0.6) +
      annotate("label",
               x = -Inf, y = Inf,
               label = stats_label,
               hjust = -0.05, vjust = 1.02,
               size = 2.8,
               color = "black", fill = "#FFFFF0",
               label.size = 0.4, label.r = unit(0.1, "lines")) +
      scale_y_continuous(expand = expansion(mult = c(0.05, 0.18))) +
      labs(title = trait,
           x = NULL, y = NULL) +
      theme_pub +
      theme(plot.title = element_text(hjust = 0))
    
    all_plots[[trait]] <- p
    
    ggsave(file.path(output_dir, paste0("segmented_", trait, ".png")), p,
           width = 6, height = 4, dpi = 600)
    
  }, error = function(e) {
    cat(sprintf("  %-20s  FAILED: %s\n", trait, e$message))
  })
}
#Breakpoint output csv file
breakpoint_results <- breakpoint_results %>% arrange(Trait)
write.csv(breakpoint_results, file.path(output_dir, "breakpoint_table.csv"),
          row.names = FALSE)
#Combined figure of all the individual plots

if (length(all_plots) > 0) {
  combined_plots <- lapply(all_plots, function(p) {
    p + theme(plot.title = element_text(size = 10),
              axis.title = element_text(size = 9),
              axis.text  = element_text(size = 7))
  })
  
  n_traits <- length(combined_plots)
  n_rows   <- ceiling(n_traits / ncol_combined)
  
  combined <- wrap_plots(combined_plots, ncol = ncol_combined)
  
  out_path <- file.path(output_dir,
                        sprintf("combined_segmented_bp%d.png", breakpoint_year))
  ggsave(out_path, combined,
         width  = ncol_combined * 3,
         height = max(6, n_rows * 2.8),
         dpi    = 600,
         limitsize = FALSE)
  cat("  ", basename(out_path), "\n", sep = "")
}
#*******************************************************************************
#***********Figure S4***********************************************************
library(readxl)
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggrepel)
library(patchwork)
library(viridis)
# Classification thresholds
beta_threshold <- 0.10
alpha          <- 0.05
# Era cutoffs (USED ONLY FOR YIELD DECOMPOSITION VISUALIZATION)
era_breaks <- c(-Inf, 1950, 1980, 2000, Inf)
era_labels <- c("pre-1950", "1951-1980", "1981-2000", "post-2000")
era_colors <- c(
  "pre-1950"  = "#CC5260",
  "1951-1980" = "#4AB5C4",
  "1981-2000" = "#DE77AE",
  "post-2000" = "#00A087"
)
theme_pub <- theme_minimal(base_size = 12) +
  theme(
    text             = element_text(face = "bold", color = "black"),
    plot.title       = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle    = element_text(size = 10, hjust = 0.5, color = "gray40"),
    axis.title       = element_text(face = "bold", size = 12),
    axis.text        = element_text(face = "bold", size = 10, color = "black"),
    legend.title     = element_text(face = "bold", size = 11),
    legend.text      = element_text(face = "bold", size = 10),
    panel.grid.minor = element_blank(),
    panel.border     = element_rect(color = "black", fill = NA, linewidth = 0.6)
  )

# ============================================================================
# PART 1 — TRADE-OFF IDENTIFICATION (statistical, from SEM outputs)
# ============================================================================
cat(strrep("=", 70), "\n")
cat("1. TRADE-OFF IDENTIFICATION (statistical classification)\n")
cat(strrep("=", 70), "\n\n")

paths    <- read.csv(file.path(sem_dir, "path_coefficients.csv"))
r2_table <- read.csv(file.path(sem_dir, "r_squared.csv"))

# Fix column name mismatch
names(paths)[names(paths) == "est_std"] <- "est.std"
df_raw <- read_excel(input_file, sheet = input_sheet)
names(df_raw) <- gsub("-", "_", names(df_raw))

model_vars  <- unique(c(paths$lhs, paths$rhs))
missing_var <- setdiff(model_vars, names(df_raw))
if (length(missing_var) > 0) {
  stop("Variables in path_coefficients.csv but not in data: ",
       paste(missing_var, collapse = ", "))
}

# Standardize the same way the SEM did
df_scaled <- df_raw
df_scaled[model_vars] <- lapply(df_scaled[model_vars], function(x) {
  x  <- as.numeric(x)
  sx <- sd(x, na.rm = TRUE)
  if (is.finite(sx) && sx > 0) as.numeric(scale(x)) else x
})

# Plot-level zero-order correlations (consistent with what the SEM "sees")
cor_mat <- cor(df_scaled[, model_vars], use = "pairwise.complete.obs")

# Augment paths with r, Pratt, magnitude, and category
paths$zero_order_r <- mapply(function(rhs, lhs) cor_mat[rhs, lhs],
                             paths$rhs, paths$lhs)
paths$pratt        <- paths$est.std * paths$zero_order_r

paths$magnitude <- dplyr::case_when(
  abs(paths$est.std) >= 0.30 ~ "Large",
  abs(paths$est.std) >= 0.10 ~ "Moderate",
  TRUE                       ~ "Small"
)
paths$significant     <- paths$pvalue < alpha
paths$sign_consistent <- sign(paths$est.std) == sign(paths$zero_order_r)

paths$category <- dplyr::case_when(
  paths$est.std < 0 & paths$zero_order_r < 0 &
    paths$significant & abs(paths$est.std) >= beta_threshold
  ~ "Genuine trade-off",
  paths$est.std < 0 & paths$zero_order_r > 0 &
    paths$significant
  ~ "Suppressor",
  paths$est.std > 0 & paths$zero_order_r < 0 &
    paths$significant
  ~ "Net suppression",
  paths$est.std > 0 & paths$zero_order_r > 0 &
    paths$significant & abs(paths$est.std) >= beta_threshold
  ~ "Synergy",
  !paths$significant ~ "Not significant",
  TRUE               ~ "Negligible"
)

paths <- paths %>%
  dplyr::arrange(dplyr::desc(category == "Genuine trade-off"), est.std)

write.csv(paths, file.path(output_dir, "tradeoff_classification.csv"),
          row.names = FALSE)

trade_offs_df <- paths %>%
  dplyr::filter(category == "Genuine trade-off") %>%
  dplyr::arrange(est.std)

if (nrow(trade_offs_df) > 0) {
  cat("  TRADE-OFFS (sorted by strength):\n")
  cat(sprintf("  %-22s  %7s  %7s  %7s  %s\n",
              "Path", "\u03B2", "r", "Pratt", "Magnitude"))
  cat("  ", strrep("-", 60), "\n", sep = "")
  for (i in seq_len(nrow(trade_offs_df))) {
    cat(sprintf("  %-22s  %+7.3f  %+7.3f  %+7.3f  %s\n",
                paste(trade_offs_df$rhs[i], "->", trade_offs_df$lhs[i]),
                trade_offs_df$est.std[i],
                trade_offs_df$zero_order_r[i],
                trade_offs_df$pratt[i],
                trade_offs_df$magnitude[i]))
  }
}

#Trade off plots
cultivar_df <- df_raw %>%
  group_by(variety_name, yor) %>%
  summarise(
    across(where(is.numeric), ~ mean(.x, na.rm = TRUE)),
    .groups = "drop"
  ) %>%
  arrange(yor) %>%
  mutate(
    era = case_when(
      yor <= 1950 ~ "pre-1950",
      yor <= 1980 ~ "1951-1980",
      yor <= 2000 ~ "1981-2000",
      TRUE        ~ "post-2000"
    )
  )

cat(sprintf("  Cultivars: %d  |  Years: %d - %d\n\n",
            nrow(cultivar_df), min(cultivar_df$yor), max(cultivar_df$yor)))

all_plots <- list()
tradeoff_summary <- data.frame()

for (i in seq_len(nrow(trade_offs_df))) {
  rhs      <- trade_offs_df$rhs[i]
  lhs      <- trade_offs_df$lhs[i]
  beta_val <- trade_offs_df$est.std[i]
  r_val    <- trade_offs_df$zero_order_r[i]
  
  if (!all(c(rhs, lhs) %in% names(cultivar_df))) next
  
  x_vec <- cultivar_df[[rhs]]
  y_vec <- cultivar_df[[lhs]]
  ok    <- complete.cases(x_vec, y_vec)
  if (sum(ok) < 5) next
  
  cult_r  <- cor(x_vec[ok], y_vec[ok])
  cult_r2 <- cult_r^2
  
  panel_label <- paste0("(",LETTERS[i],")")
  title_str   <- paste(rhs, "vs", lhs)
  
  stats_label <- paste0(
    "SEM \u03B2 = ", sprintf("%+.2f", beta_val), "\n",
    "Plot-level r = ", sprintf("%+.2f", r_val), "\n",
    "Cultivar r = ", sprintf("%+.2f", cult_r)
  )
  
  p <- ggplot(
    cultivar_df,
    aes(
      x = .data[[rhs]],
      y = .data[[lhs]],
      color = era
    )
  ) +
    
    geom_smooth(
      aes(group = 1),
      method = "lm",
      se = TRUE,
      color = "#CD0000",
      fill = "#CD0000",
      alpha = 0.12,
      linewidth = 0.7,
      linetype = "dashed",
      show.legend = FALSE
    ) +
    
    geom_point(
      size = 3.5,
      alpha = 0.85
    ) +
    
    geom_text_repel(
      aes(label = variety_name),
      size = 2.2,
      fontface = "bold",
      max.overlaps = 20,
      show.legend = FALSE
    ) +
    
    annotate(
      "label",
      x = Inf,
      y = Inf,
      label = stats_label,
      hjust = 1.05,
      vjust = 1.1,
      size = 3.0,
      fontface = "bold",
      color = "black",
      fill = "white",
      label.r = unit(0.1, "lines")
    ) +
    
    scale_color_manual(
      name = "Breeding Era",
      values = era_colors,
      drop = FALSE
    ) +
    
    labs(
      title = NULL,
      x = rhs,
      y = lhs
    ) +
    
    theme_pub +
    
    theme(
      plot.title = element_text(hjust = 0),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.background = element_blank(),
      axis.line = element_line(color = "black"),
      legend.position = "bottom"
    )
  all_plots[[title_str]] <- p
  
  fname <- sprintf("tradeoff_%s_%s_vs_%s.png", LETTERS[i], rhs, lhs)
  ggsave(file.path(output_dir, fname), p,
         width = 7, height = 7, dpi = 600, bg = "white")
  
  tradeoff_summary <- rbind(tradeoff_summary, data.frame(
    Panel       = LETTERS[i],
    Predictor   = rhs, Outcome = lhs,
    SEM_beta    = beta_val,
    Plot_r      = r_val,
    Cultivar_r  = round(cult_r, 3),
    Cultivar_R2 = round(cult_r2, 3),
    Magnitude   = trade_offs_df$magnitude[i],
    stringsAsFactors = FALSE
  ))
}

write.csv(tradeoff_summary, file.path(output_dir, "tradeoff_summary.csv"),
          row.names = FALSE)

# Combined trade-off figure
if (length(all_plots) >= 1) {
  
  n_cols <- min(4, length(all_plots))
  n_rows <- ceiling(length(all_plots) / n_cols)
  
  combined_to <-
    wrap_plots(
      all_plots,
      ncol = n_cols,
      guides = "collect"
    ) +
    
    plot_annotation(
      title = NULL,
      subtitle = NULL
    ) &
    
    theme(
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.box = "horizontal"
    )
  
  ggsave(
    file.path(output_dir, "tradeoffs_combined.png"),
    combined_to,
    width = n_cols * 4,
    height = n_rows * 4.0 + 0.6,
    dpi = 600,
    bg = "white",
    limitsize = FALSE
  )
  
  cat("  tradeoffs_combined.png\n")
}

#Yield (GCS, GYS, and TKW) decomposition by year
cultivar_df$era <- cut(cultivar_df$yor,
                       breaks = era_breaks,
                       labels = era_labels)

required <- c("GCS", "TKW", "GYS")
missing  <- setdiff(required, names(cultivar_df))

if (length(missing) > 0) {
  warning("Yield decomposition skipped - missing columns: ",
          paste(missing, collapse = ", "))
} else {

  yld <- cultivar_df %>%
    dplyr::filter(!is.na(GCS), !is.na(TKW), !is.na(GYS),
                  GCS > 0, TKW > 0, GYS > 0)
  
  if (nrow(yld) < 10) {
    warning(sprintf("Not enough complete cultivars for decomposition (n = %d).",
                    nrow(yld)))
  } else {
    
    fit_gys <- lm(log(GYS) ~ yor, data = yld)
    fit_gcs <- lm(log(GCS) ~ yor, data = yld)
    fit_tkw <- lm(log(TKW) ~ yor, data = yld)
    
    slope_gys <- coef(fit_gys)[2]
    slope_gcs <- coef(fit_gcs)[2]
    slope_tkw <- coef(fit_tkw)[2]
    
    p_gys <- summary(fit_gys)$coefficients[2, 4]
    p_gcs <- summary(fit_gcs)$coefficients[2, 4]
    p_tkw <- summary(fit_tkw)$coefficients[2, 4]

    gys_pct_per_yr <- slope_gys * 100
    gcs_pct_per_yr <- slope_gcs * 100
    tkw_pct_per_yr <- slope_tkw * 100

    gcs_contrib <- if (slope_gys != 0) slope_gcs / slope_gys * 100 else 50
    tkw_contrib <- if (slope_gys != 0) slope_tkw / slope_gys * 100 else 50

    year_span <- max(yld$yor) - min(yld$yor)
    gys_total_pct <- (exp(slope_gys * year_span) - 1) * 100
    gcs_total_pct <- (exp(slope_gcs * year_span) - 1) * 100
    tkw_total_pct <- (exp(slope_tkw * year_span) - 1) * 100
    
    
    era_means <- yld %>%
      group_by(era) %>%
      summarise(n         = dplyr::n(),
                GCS_mean  = round(mean(GCS), 1),
                TKW_mean  = round(mean(TKW), 2),
                GYS_mean  = round(mean(GYS), 3),
                .groups   = "drop")
    
    cat("  Era-level means:\n")
    for (i in seq_len(nrow(era_means))) {
      cat(sprintf("    %-12s  n = %2d   GCS = %5.1f   TKW = %5.2f   GYS = %5.3f\n",
                  era_means$era[i], era_means$n[i],
                  era_means$GCS_mean[i], era_means$TKW_mean[i],
                  era_means$GYS_mean[i]))
    }

    decomp_df <- data.frame(
      Component         = c("GCS", "TKW", "GYS"),
      Slope_log_per_yr  = round(c(slope_gcs, slope_tkw, slope_gys), 5),
      Pct_per_year      = round(c(gcs_pct_per_yr, tkw_pct_per_yr, gys_pct_per_yr), 3),
      Total_pct_change  = round(c(gcs_total_pct, tkw_total_pct, gys_total_pct), 1),
      p_value           = signif(c(p_gcs, p_tkw, p_gys), 4),
      Contribution_pct  = round(c(gcs_contrib, tkw_contrib, 100), 1)
    )
    write.csv(decomp_df,
              file.path(output_dir, "yield_decomposition.csv"),
              row.names = FALSE)
    write.csv(era_means,
              file.path(output_dir, "era_means.csv"),
              row.names = FALSE)
    
    #Breeding trajectory: GCS vs TKW
    p_traj <- ggplot(yld, aes(x = GCS, y = TKW, color = era)) +
      geom_smooth(aes(group = 1), method = "lm", se = TRUE,
                  color = "#CD0000", fill = "#CD0000", alpha = 0.1,
                  linewidth = 0.5, linetype = "dashed", show.legend = FALSE) +
      geom_point(size = 4, alpha = 0.85) +
      geom_text_repel(aes(label = paste0(variety_name, "\n(", yor, ")")),
                      size = 2, fontface = "bold",
                      max.overlaps = 20, show.legend = FALSE) +
      scale_color_manual(values = era_colors, name = "Breeding Era", drop = FALSE) +
      labs(
        title = NULL,
        x = "GCS",
        y = "TKW (g)"
      ) +
      
      annotate(
        "label",
        x = Inf,
        y = Inf,
        label = sprintf(
          "GCS: %+.3f%%/yr\nTKW: %+.3f%%/yr",
          gcs_pct_per_yr,
          tkw_pct_per_yr
        ),
        hjust = 1.05,
        vjust = 1.1,
        size = 3.5,
        fontface = "bold",
        fill = "white",
        color = "black",
        label.r = unit(0.15, "lines")
      ) +
      theme_pub +
      theme(panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(),
            panel.background = element_blank(),
            axis.line = element_line(color = "black"),legend.position = 'none')
    
    ggsave(file.path(output_dir, "breeding_trajectory.png"), p_traj,
           width = 6, height = 6, dpi = 600, bg = "white")
    yr0      <- min(yld$yor)
    gcs_anchor <- exp(coef(fit_gcs)[1] + slope_gcs * yr0)
    tkw_anchor <- exp(coef(fit_tkw)[1] + slope_tkw * yr0)
    gys_anchor <- exp(coef(fit_gys)[1] + slope_gys * yr0)
    
    yld$GCS_norm <- yld$GCS / gcs_anchor
    yld$TKW_norm <- yld$TKW / tkw_anchor
    yld$GYS_norm <- yld$GYS / gys_anchor
    
    norm_long <- yld %>%
      dplyr::select(variety_name, yor, GCS_norm, TKW_norm, GYS_norm) %>%
      tidyr::pivot_longer(cols = c(GCS_norm, TKW_norm, GYS_norm),
                          names_to = "Component", values_to = "Normalized")
    norm_long$Component <- factor(norm_long$Component,
                                  levels = c("GYS_norm", "GCS_norm", "TKW_norm"),
                                  labels = c("GYS ",
                                             "GCS ",
                                             "TKW "))
    
    p_norm <- ggplot(norm_long,
                     aes(x = yor, y = Normalized, color = Component)) +
      geom_smooth(method = "loess", span = 0.75, se = TRUE,
                  alpha = 0.1, linewidth = 1) +
      geom_point(size = 2, alpha = 0.5) +
      geom_hline(yintercept = 1.0, linetype = "dashed", color = "gray40") +
      scale_color_manual(
        values = c("GYS "    = "#27AE60",
                   "GCS "    = "#CD0000",
                   "TKW "  = "#4169E1"),
        name = NULL) +
      labs(title    = NULL,
           subtitle = NULL,
           x = "Year of Release",
           y = sprintf("Normalized Value (1842)", yr0)) +
      theme_pub +
      theme(panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(),
            panel.background = element_blank(),
            axis.line = element_line(color = "black"),
            legend.position = c(0.28, 0.12),
            legend.direction = "horizontal",
            
            legend.background = element_rect(
              fill = "white",
              color = "black",
              linewidth = 0.3
            ),
            
            legend.key = element_blank(),
            
            legend.title = element_blank(),
            
            legend.text = element_text(
              size = 8
            ))
    ggsave(file.path(output_dir, "yield_components_trajectory.png"), p_norm,
           width = 6, height = 6, dpi = 600, bg = "white")
  }
}
#Combined figure
p_A <- p_traj +
  labs(tag = "A") + guides(color = "none") +  
  theme(
    plot.tag = element_text(face = "bold", size = 18),
    plot.tag.position = c(0.02, 0.98),  legend.position = "none"
  )

p_B <- p_norm +
  labs(tag = "B") +
  theme(
    plot.tag = element_text(
      face = "bold",
      size = 20
    ),
    
    plot.tag.position = c(0.02, 0.98),
    
    legend.position = c(0.28, 0.12),
    legend.direction = "horizontal",
    
    legend.background = element_rect(
      fill = "white",
      color = "black",
      linewidth = 0.3
    ),
    
    legend.key = element_blank(),
    legend.title = element_blank(),
    
    legend.text = element_text(size = 8)
  )

# Trade-off panels
tradeoff_panels <- unname(all_plots)

# Add C–F tags
for (i in seq_along(tradeoff_panels)) {
  
  tradeoff_panels[[i]] <-
    tradeoff_panels[[i]] +
    
    labs(tag = LETTERS[i + 2]) +   # C D E F
    
    theme(
      plot.tag = element_text(
        face = "bold",
        size = 18
      ),
      plot.tag.position = c(0.02, 1.0)
    )
}

top_row <-
  p_A | p_B

bottom_row <-
  wrap_plots(
    tradeoff_panels,
    ncol = 4,
    guides = "collect"
  ) &
  theme(
    legend.position = "bottom",
    legend.direction = "horizontal"
  )

combined_final <-
  (top_row / plot_spacer() / bottom_row) +
  
  plot_layout(
    heights = c(1.2, 0.08, 1)
  ) &
  
  theme(
    plot.margin = margin(3, 3, 3, 3)
  )

theme(
  legend.position = "bottom",
  legend.direction = "horizontal",
  legend.box = "horizontal",
  
  legend.title = element_text(
    face = "bold",
    size = 10
  ),
  
  legend.text = element_text(
    size = 9
  ),
  
  plot.margin = margin(3, 3, 3, 3)
)

ggsave(
  file.path(output_dir, "combined_figure.png"),
  combined_final,
  width = 15,
  height = 9,
  dpi = 600,
  bg = "white"
)
#**********************************************************************************************************************************************************