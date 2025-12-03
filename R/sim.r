#' Simulate summary statistics from genotype data
#'
#' @param X Genotype matrix (individuals x SNPs)
#' @param af Allele frequencies
#' @param ncause Number of causal variants
#' @param sigmag Standard deviation of effect sizes
#' @param seed Random seed for reproducibility
#' @return A tibble with simulated summary statistics: betahat, b (true effect sizes), se, zhat, pval, af
#' @export
simulate_ss <- function(X, af, ncause, sigmag, seed=1234) {
    set.seed(seed)
    nsnp <- length(af)
    nid <- nrow(X)
    b <- rep(0, nsnp)
    b[sample(1:nsnp, ncause)] <- rnorm(ncause, sd=sigmag)

    e <- rnorm(nid)
    y <- X %*% b + e 

    betahat <- sapply(1:nsnp, \(i) {cov(X[,i], y) / var(X[,i])})
    se <- sapply(1:nsnp, \(i) {sqrt(var(y) / (var(X[,i] * sqrt(nid))))})
    zhat <- betahat/se
    pval <- 2 * pnorm(-abs(zhat))

    return(tibble(betahat, b, se, zhat, pval, af))
}
