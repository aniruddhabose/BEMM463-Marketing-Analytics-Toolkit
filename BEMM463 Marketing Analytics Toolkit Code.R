
# 1. INSTALL AND LOAD REQUIRED PACKAGES

install.packages(c("tidyverse", "cluster", "factoextra", "mlogit", "conjoint", 
                   "ggplot2", "ggradar", "gridExtra", "readr", "dplyr"))
library(tidyverse)    # Data manipulation & visualization
library(cluster)      # Cluster analysis
library(factoextra)   # Clustering visualization
library(mlogit)       # Choice modeling (Multinomial Logit)
library(conjoint)     # Conjoint analysis
library(ggplot2)      # Plotting
library(ggradar)      # Radar charts
library(gridExtra)    # Arrange multiple plots
library(readr)        # Reading CSV files
library(dplyr)


# 2. SELECTING AND READING THE REQUIRED FILE

df <- read_csv(file.choose())  # Choose the Dataset

# View the DataFrame 
view(df)


# 3. DATA PREPARATION 

# Convert categorical variables to factors
df <- df %>%
  mutate(across(c(DietaryPreference, PreferredBrand, PriceSensitivity, 
                  PreferredProteinContent, FlavorPreference, 
                  EcoFriendlyPackagingImportance, SocialMediaExposure,
                  PurchasedNewProduct), as.factor))

# Check structure and missing values
glimpse(df)
sum(is.na(df))


# 4. CLUSTER ANALYSIS 

# Select variables for clustering and convert selected factors to numeric
cluster_vars <- df %>%
  select(AverageMonthlySpend, PurchaseFrequencyPerMonth, 
         PriceSensitivity, EcoFriendlyPackagingImportance) %>%
  mutate(PriceSensitivity = as.numeric(PriceSensitivity),
         EcoFriendlyPackagingImportance = as.numeric(EcoFriendlyPackagingImportance))

# Standardize the data
df_scaled <- scale(cluster_vars)

# Determine optimal number of clusters
fviz_nbclust(df_scaled, kmeans, method = "wss") + 
  ggtitle("Elbow Method for Optimal Clusters")
fviz_nbclust(df_scaled, kmeans, method = "silhouette") + 
  ggtitle("Silhouette Method for Optimal Clusters")

# Perform k-means clustering with 4 clusters
set.seed(123)
kmeans_result <- kmeans(df_scaled, centers = 4, nstart = 25)

# Add cluster assignments to original data
df$Cluster <- as.factor(kmeans_result$cluster)

# Visualize clusters and assign to cluster_plot for later use
cluster_plot <- fviz_cluster(kmeans_result, data = df_scaled, 
                             geom = "point",
                             ellipse.type = "convex",
                             ggtheme = theme_bw()) +
  ggtitle("K-means Clustering Results")
print(cluster_plot)

# Cluster profiles
cluster_profiles <- df %>%
  group_by(Cluster) %>%
  summarise(
    Size = n(),
    Avg_Spend = mean(AverageMonthlySpend),
    Avg_Frequency = mean(PurchaseFrequencyPerMonth),
    Price_Sensitivity = names(which.max(table(PriceSensitivity))),
    Top_Diet = names(which.max(table(DietaryPreference))),
    Top_Brand = names(which.max(table(PreferredBrand))),
    Top_Flavor = names(which.max(table(FlavorPreference))),
    .groups = 'drop'
  )
print(cluster_profiles)


# 5. CHOICE MODEL (MULTINOMIAL LOGIT) 

# Prepare choice data
choice_data <- df %>%
  select(CustomerID, PreferredBrand, PriceSensitivity, FlavorPreference,
         PreferredProteinContent, EcoFriendlyPackagingImportance)

# Fit multinomial logit model
mnl_model <- nnet::multinom(
  PreferredBrand ~ PriceSensitivity + FlavorPreference + 
    PreferredProteinContent + EcoFriendlyPackagingImportance,
  data = choice_data
)

# Calculate predicted probabilities
pred_probs <- fitted(mnl_model) %>%
  as.data.frame() %>%
  tibble::rownames_to_column("CustomerID") %>%
  tidyr::pivot_longer(-CustomerID, names_to = "Brand", values_to = "Probability")

# Visualize choice probabilities
choice_plot <- pred_probs %>%
  group_by(Brand) %>%
  summarise(Probability = mean(Probability)) %>%
  ggplot(aes(x = reorder(Brand, Probability), y = Probability, fill = Brand)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = scales::percent(Probability, accuracy = 0.1)), 
            vjust = -0.5, size = 4) +
  scale_y_continuous(labels = scales::percent, limits = c(0, 0.5)) +
  labs(title = "Brand Choice Probabilities", 
       x = "", y = "Probability") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5, face = "bold"),
        axis.text = element_text(size = 10))
print(choice_plot)


# 7. CONJOINT ANALYSIS 

# Create full factorial design of product profiles using the factor levels from df
profiles <- expand.grid(
  Brand = levels(df$PreferredBrand),
  Price = levels(df$PriceSensitivity),
  Protein = levels(df$PreferredProteinContent),
  Flavor = levels(df$FlavorPreference),
  Packaging = levels(df$EcoFriendlyPackagingImportance)
)

# Simulate consumer ratings for each product profile
set.seed(123)
ratings <- profiles %>%
  mutate(
    BaseRating = sample(1:10, nrow(.), replace = TRUE),
    Rating = BaseRating +
      ifelse(Brand == "Beyond Meat", 2, ifelse(Brand == "Impossible Foods", 1, 0)) +
      ifelse(Price == "Low", 1, ifelse(Price == "High", -1, 0)) +
      ifelse(Packaging == "High", 1, 0),
    Rating = pmax(1, pmin(10, Rating))
  ) %>%
  select(-BaseRating)

# Convert product profiles to numeric values for conjoint analysis
profiles_numeric <- profiles %>% 
  mutate(across(everything(), ~ as.numeric(factor(.))))

# Run part-worth utility estimation using conjoint package
conjoint_results <- caPartUtilities(
  y = ratings$Rating,
  x = profiles_numeric
)
print(conjoint_results)

# Compute attribute importance
importance <- caImportance(
  y = ratings$Rating,
  x = profiles_numeric
)

# Format the importance scores into a data frame
importance_df <- data.frame(
  Attribute = colnames(profiles),
  Importance = importance
)
print(importance_df)

# Visualize attribute importance in a bar chart
conjoint_plot <- ggplot(importance_df, 
                        aes(x = reorder(Attribute, Importance), 
                            y = Importance, 
                            fill = Attribute)) +
  geom_col(width = 0.7, show.legend = FALSE) +
  geom_text(aes(label = paste0(round(Importance, 1), "%")), 
            hjust = -0.1, size = 4) +
  coord_flip() +
  expand_limits(y = c(0, max(importance_df$Importance) * 1.2)) +
  labs(title = "Product Attribute Importance",
       x = "", y = "Relative Importance (%)") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        axis.text = element_text(size = 10)) +
  scale_fill_brewer(palette = "Set2")
print(conjoint_plot)


# 6. MARKET RESPONSE MODEL 

# Fit logistic regression for market response
market_model <- glm(
  PurchasedNewProduct ~ SocialMediaExposure + InfluencerEndorsementImpact + 
    InStorePromotionResponse,
  family = binomial,
  data = df
)

# Calculate odds ratios with confidence intervals
odds_ratios <- exp(cbind(OR = coef(market_model), confint(market_model))) %>%
  as.data.frame() %>%
  tibble::rownames_to_column("Variable") %>%
  filter(Variable != "(Intercept)")

# Visualize marketing effectiveness
market_plot <- odds_ratios %>%
  ggplot(aes(x = reorder(Variable, OR), y = OR)) +
  geom_pointrange(aes(ymin = `2.5 %`, ymax = `97.5 %`), 
                  color = "darkgreen", size = 1) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
  coord_flip() +
  labs(title = "Marketing Effectiveness", 
       x = "", y = "Odds Ratio (95% CI)") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        axis.text = element_text(size = 10))
print(market_plot)


# 8. FINAL REPORT: COMBINING GRAPHS 

# Arrange all graphs into a single final report layout
final_report <- gridExtra::grid.arrange(
  cluster_plot,
  choice_plot,
  conjoint_plot,
  market_plot,
  ncol = 2,
  nrow = 2,
  top = grid::textGrob("Plant-Based Food Consumer Analysis", 
                       gp = grid::gpar(fontsize = 16, fontface = "bold"))
)

# Save the final report image
ggsave("final_report.png", final_report, width = 14, height = 10, dpi = 300)


# 9. SAVE RESULTS 

write_csv(cluster_profiles, "cluster_profiles.csv")
write_csv(pred_probs, "brand_choice_probabilities.csv")
write_csv(odds_ratios, "marketing_effectiveness.csv")
