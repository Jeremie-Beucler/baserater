% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/generate_new_ratings.R
\name{generate_typicality}
\alias{generate_typicality}
\title{Generate typicality ratings via Hugging Face (experimental)}
\usage{
generate_typicality(
  groups,
  descriptions,
  model = "meta-llama/Llama-3-70B-Instruct",
  custom_url = NULL,
  hf_token = Sys.getenv("HF_API_TOKEN"),
  n = 25,
  min_valid = ceiling(0.8 * n),
  temperature = 1,
  top_p = 1,
  max_tokens = 3,
  retries = 4,
  matrix = TRUE,
  return_raw_scores = TRUE,
  return_full_responses = FALSE,
  verbose = interactive(),
  system_prompt = default_system_prompt(),
  user_prompt_template = default_user_prompt_template()
)
}
\arguments{
\item{groups, descriptions}{Character vectors. \emph{When} \code{matrix = FALSE} they
\strong{must} be the same length.}

\item{model}{Model ID on Hugging Face (ignored if \code{custom_url} is
supplied).}

\item{custom_url}{Fully-qualified HTTPS URL of a private Inference Endpoint
or self-hosted TGI server.}

\item{hf_token}{A Hugging Face API token (see \url{https://huggingface.co/settings/tokens}). Defaults to \code{Sys.getenv("HF_API_TOKEN")}.}

\item{n}{Samples requested per retry block (>= 1).}

\item{min_valid}{Minimum numeric scores required per pair (>= 1).}

\item{temperature, top_p, max_tokens}{Generation controls.}

\item{retries}{Maximum number of \emph{additional} retry blocks.}

\item{matrix}{\code{TRUE} = cross-product, \code{FALSE} = paired.}

\item{return_raw_scores}{If \code{TRUE}, also returns the vector(s) of raw valid numeric scores.}

\item{return_full_responses}{If \code{TRUE}, also returns all raw text model outputs
(or error strings from failed attempts) for each query.}

\item{verbose}{If \code{TRUE}, prints progress: pair labels, retry counts,
running tallies, and raw model responses/errors as they occur.}

\item{system_prompt}{Prompt string for the system message. See the 'Prompting Details' section and function signature for default content and customization.}

\item{user_prompt_template}{Prompt template for the user message, with \code{{group}} and \code{{description}} placeholders. The prompt should already include any formatting tokens required by your model (e.g., special chat tags). No additional formatting is added by the function. See the 'Prompting Details' section and function signature for default content and customization.}
}
\value{
If a pair cannot reach min_valid, its mean is NA; raw invalid strings remain available when return_full_responses = TRUE.
Cross-product mode (\code{matrix = TRUE}) -> a list containing:
\itemize{
\item \code{scores}: A matrix of mean typicality scores.
\item \code{raw} (if \code{return_raw_scores = TRUE}): A matrix of lists, where each list contains the raw numeric scores for that pair.
\item \code{full_responses} (if \code{return_full_responses = TRUE}): A matrix of lists, where each list contains all raw text model outputs (or error strings) for that pair.
}
Paired mode (\code{matrix = FALSE}) -> a tibble with columns for \code{group}, \code{description}, \code{mean_score}, and additionally:
\itemize{
\item \code{raw} (if \code{return_raw_scores = TRUE}): A list-column where each element is a vector of raw numeric scores.
\item \code{full_responses} (if \code{return_full_responses = TRUE}): A list-column where each element is a character vector of all raw text model outputs (or error strings).
}
}
\description{
This function uses the Hugging Face Inference API (or a compatible endpoint)
to generate typicality ratings by querying a large language model (LLM).
It generates one or multiple ratings for each group-description pair and returns the mean score.
It can be quite slow to run depending on the API.

\strong{Important:} Before running this function, please ensure that:
\itemize{
\item You have a valid Hugging Face API token (via \code{hf_token} or the \code{HF_API_TOKEN} environment variable);
\item You have a valid Hugging Face access token and have accepted the model’s license on the Hub;
\item The specified model is available and accessible via the Hugging Face API or your own hosted inference endpoint;
\item The model supports free-text input and generates numeric outputs in response to structured prompts.
}

Calls to the API are rate-limited, may incur usage costs, and require an internet connection.
This feature is \strong{experimental} and is not guaranteed to work with all Hugging Face-hosted models.
}
\section{Get Typicality Ratings from Hugging Face Models}{
\strong{generate_typicality()} sends structured prompts to any text-generation model
hosted on the Hugging Face Inference API (or a self-hosted endpoint) and
collects \emph{numeric} ratings (0–100) of how well a \emph{description} (e.g., an
adjective) fits a \emph{group} (e.g., an occupation). Responses that cannot be
parsed into numbers are discarded.
\subsection{Modes}{
\itemize{
\item \strong{Cross-product} (\code{matrix = TRUE}, \emph{default})    Rate every combination of
the \emph{unique} \code{groups} and \code{descriptions}. Returns a list containing matrices.
\item \strong{Paired} (\code{matrix = FALSE})                     Rate the pairs row-by-row
(\code{length(groups) == length(descriptions)}). Returns a tibble.
}

Each pair is queried repeatedly until at least \strong{\code{min_valid}} clean scores
are obtained or the retry budget is exhausted. One \emph{retry block} consists of
\strong{\code{n}} new samples; invalid or out-of-range answers are silently dropped.
}
}

\section{Prompting Details}{

The function constructs the final prompt sent to the model by concatenating the \code{system_prompt}
and the rendered \code{user_prompt_template} (where \code{{group}} and \code{{description}}
are substituted with the actual values), separated by two newlines.

The default \code{system_prompt} is:

\if{html}{\out{<div class="sourceCode">}}\preformatted{You are expert at accurately reproducing the stereotypical associations humans make,
in order to annotate data for experiments.
Your focus is to capture common societal perceptions and stereotypes,
rather than factual attributes of the groups, even when they are negative or unfounded.
}\if{html}{\out{</div>}}

The default \code{user_prompt_template} is:

\if{html}{\out{<div class="sourceCode">}}\preformatted{Rate how well the description "\{description\}" reflects the prototypical member of the group "\{group\}" on a scale from 0 ("Not at all") to 100 ("Extremely").

To clarify, consider the following examples:
1. "Rate how well the description "FUNNY" reflects the prototypical member of the group "CLOWN" on a scale from 0 (Not at all) to 100 (Extremely)." A high rating is expected because "FUNNY" closely aligns with typical characteristics of a "CLOWN".
2. "Rate how well the description "FEARFUL" reflects the prototypical member of the group "FIREFIGHTER" on a scale from 0 (Not at all) to 100 (Extremely)." A low rating is expected because "FEARFUL" diverges from typical characteristics of a "FIREFIGHTER".
3. "Rate how well the description "PATIENT" reflects the prototypical member of the group "ENGINEER" on a scale from 0 (Not at all) to 100 (Extremely)." A mid-scale rating is expected because "PATIENT" neither strongly aligns with nor diverges from typical characteristics of an "ENGINEER".

Your response should be a single score between 0 and 100, with no additional text, letters, or symbols.
}\if{html}{\out{</div>}}

You are responsible for ensuring that the combination of these prompts, \\
or your custom prompts, includes any specific formatting tokens required \\
by your model (e.g., instruction tags, chat role indicators like \verb{[INST]}, \\
\verb{<|user|>}, etc.). The function itself only performs the concatenation \\
described above.

Rate-limit friendliness: transient HTTP 429/5xx errors are retried \\
(exponential back-off), and \code{wait_for_model = TRUE} is set so the call \\
blocks until the model is ready.
}

\examples{
\dontrun{
# --- Minimal reproducible example (toy input) ---
toy_groups <- c("engineer", "clown", "firefighter")
toy_descriptions <- c("patient", "funny", "fearful")

toy_result <- generate_typicality(
  groups = toy_groups,
  descriptions = toy_descriptions,
  model = "meta-llama/Llama-3-70B-Instruct",
  n = 10,
  min_valid = 8, # at least 8 valid scores per pair, we take the mean of those
  matrix = FALSE,
  return_raw_scores = TRUE,
  return_full_responses = FALSE,
  verbose = TRUE
)

print(toy_result)

# --- Full-scale example using the validation ratings ---
# ratings <- download_data("validation_ratings")

# new_scores <- generate_typicality(
#   groups                = ratings$group,
#   descriptions          = ratings$adjective,
#   model                 = "meta-llama/Llama-3.1-8B-Instruct",
#   n                     = 25,
#   min_valid             = 20,
#   max_tokens            = 5,
#   retries               = 1,
#   matrix                = FALSE,
#   return_raw_scores     = TRUE,
#   return_full_responses = TRUE,
#   verbose               = TRUE
# )

# head(new_scores)
}
}
