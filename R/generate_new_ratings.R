#' Generate typicality ratings via an Inference Provider (experimental)
#'
#' @description
#' This function uses a compatible Inference Provider API (e.g., Together AI or Fireworks)
#' to generate typicality ratings by querying a large language model (LLM).
#' It generates one or multiple ratings for each group-description pair and returns the mean score.
#' It can be quite slow to run depending on the API.
#'
#' **Important:** Before running this function, please ensure that:
#' - You have a valid API token from your inference provider (via `api_token` or an environment variable);
#' - You have provided the correct and complete URL for the provider's chat completions endpoint;
#' - The specified model is available and accessible via the endpoint;
#' - The model supports the standard `messages` array format (with system/user roles) and generates numeric outputs in response to the prompts.
#'
#' Calls to the API are rate-limited, may incur usage costs, and require an internet connection.
#' This feature is **experimental** and is not guaranteed to work with all models or providers.
#'
#' # Get Typicality Ratings from Large Language Models
#'
#' **generate_typicality()** sends structured prompts to any text-generation model
#' served via an compatible API endpoint and collects *numeric* ratings (0-100)
#' of how well a *description* (e.g., an adjective) fits a *group* (e.g., an
#' occupation). Responses that cannot be parsed into numbers are discarded.
#'
#' ## Modes
#' * **Cross-product** (`matrix = TRUE`, *default*)    Rate every combination of
#'   the *unique* `groups` and `descriptions`. Returns a list containing matrices.
#' * **Paired** (`matrix = FALSE`)                     Rate the pairs row-by-row
#'   (`length(groups) == length(descriptions)`). Returns a tibble.
#'
#' Each pair is queried repeatedly until at least **`min_valid`** clean scores
#' are obtained or the retry budget is exhausted. One *retry block* consists of
#' **`n`** new samples; invalid or out-of-range answers are silently dropped.
#'
#' @section Prompting Details:
#' The function constructs a `messages` array for the API request.
#' The `system_prompt` becomes the content of the `system` role message, and the
#' rendered `user_prompt_template` (where `{group}` and `{description}`
#' are substituted with the actual values) becomes the content of the `user` role message.
#'
#' The default `system_prompt` is:
#'
#' ```
#' You are expert at accurately reproducing the stereotypical associations
#' humans make, in order to annotate data for experiments.
#' Your focus is to capture common societal perceptions and stereotypes,
#' rather than factual attributes of the groups,
#' even when they are negative or unfounded.
#' ```
#'
#' The default `user_prompt_template` is:
#'
#' ```
#' Rate how well the description "{description}" reflects the prototypical
#' member of the group "{group}" on a scale from 0 ("Not at all") to 100
#' ("Extremely").
#'
#' To clarify, consider the following examples:
#' 1. "Rate how well the description "FUNNY" reflects the prototypical member
#'    of the group "CLOWN" on a scale from 0 (Not at all) to 100 (Extremely)."
#'    A high rating is expected because "FUNNY" closely aligns with typical
#'    characteristics of a "CLOWN".
#' 2. "Rate how well the description "FEARFUL" reflects the prototypical member
#'    of the group "FIREFIGHTER" on a scale from 0 (Not at all) to 100
#'    (Extremely)." A low rating is expected because "FEARFUL" diverges from
#'    typical characteristics of a "FIREFIGHTER".
#' 3. "Rate how well the description "PATIENT" reflects the prototypical member
#'    of the group "ENGINEER" on a scale from 0 (Not at all) to 100
#'    (Extremely)." A mid-scale rating is expected because "PATIENT" neither
#'    strongly aligns with nor diverges from typical characteristics of an
#'    "ENGINEER".
#'
#' Your response should be a single score between 0 and 100, with no additional
#' text, letters, or symbols.
#' ```
#'
#' Rate-limit friendliness: transient HTTP 429/5xx errors are retried
#' (exponential back-off).
#'
#' @param groups,descriptions   Character vectors. *When* `matrix = FALSE` they
#'   **must** be the same length.
#' @param api_url       Fully-qualified HTTPS URL for the provider's chat completions endpoint (e.g., "https://api.together.xyz/v1/chat/completions").
#' @param api_token     API token for the inference provider.
#' @param model         Model identifier string to be passed in the API request body. Check your provider's documentation for the available models and correct names.
#' @param n             Samples requested per retry block (>= 1).
#' @param min_valid     Minimum numeric scores required per pair (>= 1).
#' @param temperature,top_p,max_tokens  Generation controls.
#' @param retries       Maximum number of *additional* retry blocks.
#' @param matrix        `TRUE` = cross-product, `FALSE` = paired.
#' @param return_raw_scores  If `TRUE`, also returns the vector(s) of raw valid numeric scores.
#' @param return_full_responses If `TRUE`, also returns all raw text model outputs
#'   (or error strings from failed attempts) for each query.
#' @param verbose       If `TRUE`, prints progress: pair labels, retry counts,
#'   running tallies, and raw model responses/errors as they occur.
#' @param system_prompt Prompt string for the system message. See the 'Prompting Details' section and function signature for default content and customization.
#' @param user_prompt_template Prompt template for the user message, with `{group}` and `{description}` placeholders. No additional formatting is added by the function. See the 'Prompting Details' section and function signature for default content and customization.
#'
#' @return
#'   If a pair cannot reach min_valid, its mean is NA; raw invalid strings remain available when return_full_responses = TRUE.
#'   Cross-product mode (`matrix = TRUE`) -> a list containing:
#'   \itemize{
#'     \item `scores`: A matrix of mean typicality scores.
#'     \item `raw` (if `return_raw_scores = TRUE`): A matrix of lists, where each list contains the raw numeric scores for that pair.
#'     \item `full_responses` (if `return_full_responses = TRUE`): A matrix of lists, where each list contains all raw text model outputs (or error strings) for that pair.
#'   }
#'   Paired mode (`matrix = FALSE`) -> a tibble with columns for `group`, `description`, `mean_score`, and additionally:
#'   \itemize{
#'     \item `raw` (if `return_raw_scores = TRUE`): A list-column where each element is a vector of raw numeric scores.
#'     \item `full_responses` (if `return_full_responses = TRUE`): A list-column where each element is a character vector of all raw text model outputs (or error strings).
#'   }
#' @export
#' @examples
#' \dontrun{
#' # --- Minimal reproducible example (toy input) ---
#' # Ensure you have set your API URL and Token, e.g.:
#' # Sys.setenv(PROVIDER_API_URL = "https://api.together.xyz/v1/chat/completions")
#' # Sys.setenv(PROVIDER_API_TOKEN = "your_secret_token_here")
#'
#' toy_groups <- c("engineer", "clown", "firefighter")
#' toy_descriptions <- c("patient", "funny", "fearful")
#'
#' toy_result <- generate_typicality(
#'   groups = toy_groups,
#'   descriptions = toy_descriptions,
#'   api_url = Sys.getenv("PROVIDER_API_URL"),
#'   api_token = Sys.getenv("PROVIDER_API_TOKEN"),
#'   model = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
#'   n = 10,
#'   min_valid = 8, # at least 8 valid scores per pair, we take the mean of those
#'   matrix = FALSE,
#'   return_raw_scores = TRUE,
#'   return_full_responses = FALSE,
#'   verbose = TRUE
#' )
#'
#' print(toy_result)
#'
#' # --- Full-scale example using the validation ratings ---
#' # ratings <- download_data("validation_ratings")
#'
#' # new_scores <- generate_typicality(
#' #   groups                = ratings$group,
#' #   descriptions          = ratings$adjective,
#' #   api_url               = Sys.getenv("PROVIDER_API_URL"),
#' #   api_token             = Sys.getenv("PROVIDER_API_TOKEN"),
#' #   model                 = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
#' #   n                     = 25,
#' #   min_valid             = 20,
#' #   max_tokens            = 5,
#' #   retries               = 1,
#' #   matrix                = FALSE,
#' #   return_raw_scores     = TRUE,
#' #   return_full_responses = TRUE,
#' #   verbose               = TRUE
#' # )
#'
#' # head(new_scores)
#' }
generate_typicality <- function(
    groups,
    descriptions,
    api_url,
    api_token,
    model              = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    n                  = 25,
    min_valid          = ceiling(0.8 * n),
    temperature        = 1,
    top_p              = 1,
    max_tokens         = 3,
    retries            = 4,
    matrix             = TRUE,
    return_raw_scores  = TRUE,
    return_full_responses = FALSE,
    verbose            = interactive(),
    system_prompt = default_system_prompt(),
    user_prompt_template = default_user_prompt_template()) {

  if (missing(api_url) || !is.character(api_url) || api_url == "")
    stop("API endpoint URL not found. Please provide a valid URL via the `api_url` argument.", call. = FALSE)
  if (missing(api_token) || !is.character(api_token) || api_token == "")
    stop("API token not found. Please provide it via the `api_token` argument.", call. = FALSE)


  ## ---- helpers ----
  parse_num <- function(x) {
    if (is.na(x) || !is.character(x) || x == "") return(NA_real_)
    v <- suppressWarnings(as.numeric(gsub("[^0-9.]", "", x)))
    if (!is.na(v) && v >= 0 && v <= 100) v else NA_real_
  }

  build_user_prompt <- function(g, d) {
    glue::glue(user_prompt_template, description = d, group = g)
  }

  perform_call <- function(user_prompt_text) {
    messages <- list(
      list(role = "system", content = system_prompt),
      list(role = "user", content = user_prompt_text)
    )

    body <- list(
      model = model,
      messages = messages,
      max_tokens = max_tokens,
      temperature = temperature,
      top_p = if (top_p < 1) top_p else NULL,
      n = n,
      stream = FALSE
    )

    httr2::request(api_url) |>
      httr2::req_headers(
        Authorization = paste("Bearer", api_token),
        `Content-Type` = "application/json"
      ) |>
      httr2::req_body_json(body) |>
      httr2::req_timeout(60) |>
      httr2::req_retry(
        max_tries = 2, # 1 initial + 2 retries by httr2 for transient errors
        is_transient = \(r) httr2::resp_status(r) %in% c(429, 500, 502, 503, 504),
        backoff = ~stats::runif(1, min = 1, max = 5)
      ) |>
      httr2::req_perform()
  }

  ## ---- grid ----
  if (matrix) {
    unique_grps <- unique(groups)
    unique_dscs <- unique(descriptions)
    processing <- tidyr::crossing(group = unique_grps, description = unique_dscs)
    score_mat  <- matrix(NA_real_, length(unique_grps), length(unique_dscs),
                         dimnames = list(unique_grps, unique_dscs))
    if (return_raw_scores) {
      raw_mat <- matrix(vector("list", length(score_mat)), nrow = nrow(score_mat), dimnames = dimnames(score_mat))
    }
    if (return_full_responses) {
      full_responses_mat <- matrix(vector("list", length(score_mat)), nrow = nrow(score_mat), dimnames = dimnames(score_mat))
    }
  } else {
    if (length(groups) != length(descriptions)) {
      stop("When matrix = FALSE, 'groups' and 'descriptions' must have the same length.", call. = FALSE)
    }
    processing <- tibble::tibble(group = groups, description = descriptions)
    score_vec  <- rep(NA_real_, nrow(processing))
    if (return_raw_scores) raw_vec <- vector("list", nrow(processing))
    if (return_full_responses) full_responses_vec <- vector("list", nrow(processing))
  }

  total_pairs <- nrow(processing)
  if (verbose) {
    message(sprintf("Starting generate_typicality - %d pairs (%d samples per block, min %d valid scores per pair, up to %d retry blocks)",
                    total_pairs, n, min_valid, retries))
  }

  ## ---- main loop ----
  for (idx in seq_len(total_pairs)) {
    g <- processing$group[idx]; d <- processing$description[idx]
    if (verbose) message(sprintf("[%d/%d] Processing pair: %s x %s", idx, total_pairs, g, d))

    collected_good_scores  <- numeric(0)
    all_raw_texts_this_pair <- if (return_full_responses) character(0) else NULL

    current_pair_total_api_attempts <- 0

    # Loop for initial block (0) + specified number of retries
    for (try_block_num in 0:retries) {
      if (length(collected_good_scores) >= min_valid) break # Already have enough scores

      if (verbose) {
        if (try_block_num == 0) {
          message(sprintf("  - Initial sampling block (up to %d samples)", n))
        } else {
          message(sprintf("  - Retry block %d of %d (up to %d samples)", try_block_num, retries, n))
        }
      }

      current_pair_total_api_attempts <- current_pair_total_api_attempts + 1
      raw_texts_this_block <- character(0)
      parsed_scores_this_block <- numeric(0)

      current_user_prompt <- build_user_prompt(g, d)
      resp_obj <- try(perform_call(current_user_prompt), silent = TRUE)

      if (inherits(resp_obj, "try-error")) {
        error_message <- as.character(resp_obj)
        error_message_oneline <- gsub("\n", " ", error_message)
        raw_texts_this_block <- paste0("API_CALL_ERROR: ", error_message_oneline)
        if (verbose) {
          message(sprintf("    API call failed. Error: %s", error_message_oneline))
        }
      } else {
        status_code <- httr2::resp_status(resp_obj)
        if (status_code >= 300) {
          error_body_text <- tryCatch({
            httr2::resp_body_string(resp_obj)
          }, error = function(e_body) {
            paste("Failed to retrieve error body:", as.character(e_body))
          })
          error_body_oneline <- gsub("\n", " ", error_body_text)
          raw_texts_this_block <- paste0("HTTP_ERROR: ", status_code, " Body: ", error_body_oneline)
          if (verbose) {
            message(sprintf("    HTTP error %d. Response: %s", status_code, error_body_oneline))
          }
        } else {
          parsed_json_body <- try(httr2::resp_body_json(resp_obj), silent = TRUE)

          if (inherits(parsed_json_body, "try-error")) {
            json_error_oneline <- gsub("\n", " ", as.character(parsed_json_body))
            raw_texts_this_block <- paste0("JSON_PARSE_ERROR: ", json_error_oneline)
            if (verbose) {
              message(sprintf("    Failed to parse JSON. Error: %s", json_error_oneline))
            }
          } else {
            if (!is.list(parsed_json_body) || is.null(parsed_json_body$choices)) {
              raw_texts_this_block <- "MODEL_RESPONSE_MALFORMED: Missing or invalid 'choices' field."
              if (verbose) message("    Model response malformed or missing 'choices'.")
            } else {
              # Extract multiple completions from parsed_json_body$choices
              for (choice in parsed_json_body$choices) {
                if (!is.null(choice$message$content)) {
                  model_generated_text <- choice$message$content
                  raw_texts_this_block <- c(raw_texts_this_block, model_generated_text)

                  if (verbose && return_full_responses) {
                    message(sprintf("    Model Response: \"%s\"", model_generated_text))
                  }

                  parsed_num_val <- parse_num(model_generated_text)
                  if (!is.na(parsed_num_val)) {
                    parsed_scores_this_block <- c(parsed_scores_this_block, parsed_num_val)
                  } else if (verbose) {
                    message(sprintf("    (Response not parsable to valid score 0-100)"))
                  }
                }
              }
            }
          }
        }
      }

      collected_good_scores <- c(collected_good_scores, parsed_scores_this_block)
      if (return_full_responses) {
        all_raw_texts_this_pair <- c(all_raw_texts_this_pair, raw_texts_this_block)
      }

      if (verbose) {
        message(sprintf("  - Block summary: %d valid scores obtained this block. Total valid for pair: %d / %d.",
                        length(parsed_scores_this_block), length(collected_good_scores), min_valid))
      }
    } # End loop for retry blocks (try_block_num)

    mean_val <- if (length(collected_good_scores) >= min_valid) mean(collected_good_scores) else NA_real_

    if (matrix) {
      score_mat[g, d] <- mean_val
      if (return_raw_scores) raw_mat[[g, d]] <- collected_good_scores
      if (return_full_responses) full_responses_mat[[g, d]] <- all_raw_texts_this_pair
    } else {
      score_vec[idx] <- mean_val
      if (return_raw_scores) raw_vec[[idx]] <- collected_good_scores
      if (return_full_responses) full_responses_vec[[idx]] <- all_raw_texts_this_pair
    }
  }  # end for idx (main loop over pairs)

  if (verbose) {
    done_count <- if (matrix) sum(!is.na(score_mat)) else sum(!is.na(score_vec))
    message(sprintf("Finished - %d / %d pairs have a calculated mean score.", done_count, total_pairs))
  }

  if (matrix) {
    out <- list(scores = score_mat)
    if (return_raw_scores) out$raw <- raw_mat
    if (return_full_responses) out$full_responses <- full_responses_mat
  } else {
    out_tibble <- dplyr::mutate(processing, mean_score = score_vec)
    if (return_raw_scores) out_tibble$raw_scores <- raw_vec # Use distinct name
    if (return_full_responses) out_tibble$full_responses <- full_responses_vec
    out <- out_tibble
  }
  out
}

# Internal helper - default system prompt
default_system_prompt <- function() {
  "You are expert at accurately reproducing the stereotypical associations humans make, in order to annotate data for experiments.\nYour focus is to capture common societal perceptions and stereotypes, rather than factual attributes of the groups, even when they are negative or unfounded."
}

# Internal helper - default user-prompt template
default_user_prompt_template <- function() {
  'Rate how well the description "{description}" reflects the prototypical member of the group "{group}" on a scale from 0 ("Not at all") to 100 ("Extremely").\n\nTo clarify, consider the following examples:\n1. "Rate how well the description "FUNNY" reflects the prototypical member of the group "CLOWN" on a scale from 0 (Not at all) to 100 (Extremely)." A high rating is expected because "FUNNY" closely aligns with typical characteristics of a "CLOWN".\n2. "Rate how well the description "FEARFUL" reflects the prototypical member of the group "FIREFIGHTER" on a scale from 0 (Not at all) to 100 (Extremely)." A low rating is expected because "FEARFUL" diverges from typical characteristics of a "FIREFIGHTER".\n3. "Rate how well the description "PATIENT" reflects the prototypical member of the group "ENGINEER" on a scale from 0 (Not at all) to 100 (Extremely)." A mid-scale rating is expected because "PATIENT" neither strongly aligns with nor diverges from typical characteristics of an "ENGINEER".\n\nYour response should be a single score between 0 and 100, with no additional text, letters, or symbols.'
}
