Getting started with the baserater package
================

The `baserater` package allows to:

- Download LLM‑generated base-rate item datasets and human validation
  ratings from the original paper.  
- Generate new typicality scores with any Hugging Face model.  
- Benchmark new scores against human ground truth, and compare
  performance against strong LLM baselines.  
- Build base‑rate items database from typicality matrices.

It is designed to streamline the creation of base-rate neglect items for
reasoning experiments. You can use it to generate new typicality ratings
with any LLM on the Hugging Face platform using your own prompts and
parameters, benchmark these ratings against human data, and construct a
base-rate item dataset from the generated scores.

To learn more about the theoretical framework and studies underlying the
baserater package, see the paper: *Using Large Language Models to
Estimate Belief Strength in Reasoning* (Beucler et al., forthcoming).

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
new_scores <- generate_typicality(
  groups       = c("nurse", "clown"),
  descriptions = c("caring", "funny"),
  model        = "meta-llama/Llama-3.1-8B-Instruct",
  hf_token     = "your_token_here",
  n            = 3,
  min_valid    = 2,
  matrix       = FALSE
)
```

**Note:** To use `generate_typicality()`, you must have a [Hugging
Face](https://huggingface.co) account and an access token.  
You can create one [here](https://huggingface.co/settings/tokens).

## Evaluate model predictions

``` r
evaluate_external_ratings(new_scores)
```

## Build a base-rate item dataset

``` r
gpt4_matrix    <- download_data("typicality_matrix_gpt4")
base_rate_tbl  <- extract_base_rate_items(gpt4_matrix)
```

## More

Full documentation: <https://jeremie-beucler.github.io/baserater/>

Original paper: *Using Large Language Models to Estimate Belief Strength
in Reasoning* (Beucler et al., forthcoming).

## License

GPL-3

## Hugging Face Resources

Hugging Face is a platform that provides access to a wide range of
pre-trained models and datasets for natural language processing (NLP)
tasks. The `baserater` package uses Hugging Face’s API to generate
typicality scores with various models. Here are some useful resources to
get started with Hugging Face:

- [Create a Hugging Face access
  token](https://huggingface.co/settings/tokens) — Required to
  authenticate API requests to hosted models.  
- [Hugging Face Inference
  Endpoints](https://huggingface.co/docs/inference-endpoints) — Set up
  and deploy your own scalable model endpoint directly through Hugging
  Face.  
- [Hugging Face Inference
  Providers](https://huggingface.co/docs/inference-providers) — Use
  third-party infrastructure (e.g., AWS, Azure, Paperspace) to serve and
  scale models.  
- [Hugging Face Model Hub](https://huggingface.co/models) — Browse
  available models (e.g., LLaMA, Mixtral) and view license requirements.
