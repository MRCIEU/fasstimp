library(glue)
library(dplyr)
library(here)

dir.create("inst/extdata/1kg", showWarnings = FALSE)

plink <- "~/bin/plink2"
bfile <- "~/repo/opengwas-api-internal/opengwas-api/app/ld_files/EUR"

glue("{plink} --bfile {bfile} --r-unphased square ref-based --keep-allele-order --out inst/extdata/1kg/region --threads 8 --chr 2 --from-bp 201233443 --to-bp 201861836") %>% system()

ld <- read.table("inst/extdata/1kg/region.unphased.vcor1") %>% as.matrix()
dim(ld)

pc <- eigen(ld)
saveRDS(pc, "inst/extdata/1kg/region_ldpc.rds")

system("gzip -f inst/extdata/1kg/region.unphased.vcor1")

glue("{plink} --bfile {bfile} --recode A --keep-allele-order --out inst/extdata/1kg/region --threads 8 --chr 2 --from-bp 201233443 --to-bp 201861836") %>% system()

mat <- read.table("inst/extdata/1kg/region.raw", header=TRUE)
X <- as.matrix(mat[, -(1:6)])
X[1:5, 1:5]

map <- do.call(rbind, strsplit(colnames(X), "_")) %>% as_tibble()
save(X, map, file="inst/extdata/1kg/region_geno.rdata")



