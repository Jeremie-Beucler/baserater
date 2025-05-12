Getting started with the baserater package
================

The `baserater` package allows to:

- Download LLM‑generated base-rate item datasets and human validation
  ratings from the original paper.  
- Generate new typicality scores with any Hugging Face model.  
- Benchmark new scores against human ground truth, and compare
  performance against strong LLM baselines.  
- Build base‑rate items database from typicality matrices.

To learn more about the theoretical framework, modeling rationale, and
validation studies underlying the baserater package, see our forthcoming
paper:

**Using Large Language Models to Estimate Belief Strength in
Reasoning**  
(Beucler et al., forthcoming)

## Installation

``` r
# development version
# install.packages("pak")
pak::pak("Jeremie-Beucler/baserater")
```

## Download data

``` r
library(tidyverse)
library(baserater)

database <- download_data("database")             # full base‑rate item database
ratings   <- download_data("validation_ratings")  # 100 human‑rated items
```

## Generate scores with an LLM

``` r
new_scores <- hf_typicality(
  groups       = c("nurse", "clown"),
  descriptions = c("caring", "funny"),
  model        = "meta-llama/Llama-3.1-8B-Instruct",
  hf_token     = "your_token_here",
  n            = 3,
  min_valid    = 2,
  matrix       = FALSE
)
```

## Evaluate model predictions

``` r
# load new precomputed new scores
new_scores <- readRDS(system.file("extdata", "new_typicality_scores_llama3.1_8B.rds", package = "baserater"))

new_scores <- new_scores %>% 
  mutate(adjective = description,
         rating = mean_score) %>%
  select(group, adjective, rating)

evaluate_external_ratings(new_scores)
```

## Build a base-rate item dataset

``` r
gpt4_matrix    <- download_data("typicality_matrix_gpt4")
base_rate_tbl  <- extract_base_rate_items(gpt4_matrix)
```

## More

Full documentation: <https://jeremie-beucler.github.io/baserater/>

Paper: *Using Large Language Models to Estimate Belief Strength in
Reasoning* (Beucler et al., forthcoming)

## License

GPL-3
