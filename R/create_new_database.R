#' Extract Base-Rate Items from a Typicality Matrix
#'
#' This function processes a typicality matrix to identify base-rate items
#' by comparing typicality scores of descriptions between all unique pairs of groups.
#'
#' For each pair of groups and each description (e.g., adjective), it identifies which group
#' received the higher typicality score. The output includes the names of both groups, their scores,
#' and the log-ratio between the higher and lower score.
#'
#' By construction, the returned `Group1` always has a higher or equal typicality score
#' than `Group2` for a given description. This ensures that the resulting `StereotypeStrength`
#' (defined as `log(Score1 / Score2)`) is always **positive or zero**, and represents the strength
#' of the stereotypical association in favor of `Group1`.
#'
#' @param typicality_matrix A numeric matrix or data frame where rows are groups
#'        and columns are descriptions. If a data frame, the first column is assumed to contain
#'        the group names.
#'
#' @return A data frame with the following columns:
#' \describe{
#'   \item{Group1}{The group with the higher typicality score for the description.}
#'   \item{Group2}{The group with the lower typicality score.}
#'   \item{Description}{The description (e.g., adjective) being compared.}
#'   \item{Score1}{The typicality score for Group1.}
#'   \item{Score2}{The typicality score for Group2.}
#'   \item{StereotypeStrength}{The log-ratio: \code{log(Score1 / Score2)}. Always >= 0.}
#' }
#' @export
#' @examples
#' mat <- matrix(runif(9, 1, 100), nrow = 3,
#'               dimnames = list(c("GroupA", "GroupB", "GroupC"),
#'                               c("smart", "brave", "greedy")))
#' extract_base_rate_items(mat)

extract_base_rate_items <- function(typicality_matrix) {
  # Load necessary package for column_to_rownames
  if (!requireNamespace("tibble", quietly = TRUE)) {
    stop("The 'tibble' package is required but not installed.")
  }

  # If data frame and first column is group, convert to rownames
  if (is.data.frame(typicality_matrix)) {
    typicality_matrix <- tibble::column_to_rownames(typicality_matrix, var = colnames(typicality_matrix)[1])
  }

  # Attempt to coerce to matrix
  if (!is.matrix(typicality_matrix)) {
    typicality_matrix <- as.matrix(typicality_matrix)
  }

  # Try to convert all entries to numeric
  suppressWarnings(storage.mode(typicality_matrix) <- "numeric")

  if (any(is.na(typicality_matrix))) {
    stop("The typicality matrix contains non-numeric values that could not be coerced to numeric")
  }

  # Transpose to match expected format (groups as rows, descriptions as columns)
  typicality_matrix <- t(typicality_matrix)

  # Check that rownames and colnames exist
  if (is.null(rownames(typicality_matrix)) || is.null(colnames(typicality_matrix))) {
    stop("The matrix must have both row names (descriptions) and column names (groups)")
  }

  descriptions <- rownames(typicality_matrix)
  groups <- colnames(typicality_matrix)

  if (length(groups) < 2) {
    stop("The matrix must contain at least two groups (columns)")
  }
  if (length(descriptions) < 1) {
    stop("The matrix must contain at least one description (row)")
  }

  epsilon <- 1e-6  # Small constant to avoid division by zero if score is exactly zero

  result_list <- list()

  for (i in 1:(length(groups) - 1)) {
    for (j in (i + 1):length(groups)) {
      group1_name <- groups[i]
      group2_name <- groups[j]

      for (desc in descriptions) {
        score1_raw <- typicality_matrix[desc, group1_name]
        score2_raw <- typicality_matrix[desc, group2_name]

        # Add small offset only if score is zero
        score1 <- ifelse(score1_raw == 0, epsilon, score1_raw)
        score2 <- ifelse(score2_raw == 0, epsilon, score2_raw)

        if (score1 >= score2) {
          g1 <- group1_name
          g2 <- group2_name
          s1 <- score1
          s2 <- score2
        } else {
          g1 <- group2_name
          g2 <- group1_name
          s1 <- score2
          s2 <- score1
        }

        stereotype_strength <- log(s1 / s2)

        result_list[[length(result_list) + 1]] <- data.frame(
          Group1 = g1,
          Group2 = g2,
          Description = desc,
          Score1 = s1,
          Score2 = s2,
          StereotypeStrength = stereotype_strength,
          stringsAsFactors = FALSE
        )
      }
    }
  }

  do.call(rbind, result_list)
}
