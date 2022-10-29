library(tidyverse)
library(corrplot)
library(Rtsne)
library(umap)

setwd("/Users/Yaqian/Study/UCLA/Workshop")
#work flow: Preprocessing → normalization → t-SNE→ classification algorithm

# Section 0: loading & cleaning dataset ####
# Dataset: acoustic
df.aco <- read_csv("acoustic.csv")

# number of tokens associated with a particular voice type
unique(df.aco$Filename[df.aco$type=='fry'])#638
unique(df.aco$Filename[df.aco$type=='pd'])#3273
unique(df.aco$Filename[df.aco$type=='modal'])#1575

# remove str/shrf0; but you can keep the f0s if you want to play with it
df.aco <- df.aco[!grepl('shrF0',colnames(df.aco))] # remove shrF0
df.aco <- df.aco[!grepl('strF0',colnames(df.aco))] # remove strF0

# binary code gender
df.aco$gender_code[df.aco$gender=='female'] <- 0
df.aco$gender_code[df.aco$gender=='male'] <- 1

glimpse(df.aco)

##
# Dataset: acoustic w/ time info (9 timepoint)
df.aco.9p <- read_csv("acoustic_9p.csv")
df.aco.9p <- df.aco.9p[!grepl('u',colnames(df.aco.9p))] # remove uncorrected formants
df.aco.9p <- df.aco.9p[!grepl('shrF0',colnames(df.aco.9p))] # remove shrF0
df.aco.9p <- df.aco.9p[!grepl('strF0',colnames(df.aco.9p))] # remove strF0
df.aco.9p <- df.aco.9p[!grepl('sF0',colnames(df.aco.9p))] # remove sF0
df.aco.9p <- df.aco.9p[!grepl('pF1',colnames(df.aco.9p))] # remove praatF1
df.aco.9p <- df.aco.9p[!grepl('pF2',colnames(df.aco.9p))] # remove pF2
df.aco.9p <- df.aco.9p[!grepl('pF3',colnames(df.aco.9p))] # remove pF3
df.aco.9p <- df.aco.9p[!grepl('pF4',colnames(df.aco.9p))] # remove pF4
df.aco.9p <- df.aco.9p[!grepl('pB',colnames(df.aco.9p))] # remove pBx

df.aco.9p$gender_code[df.aco.9p$gender=='female'] <- 0
df.aco.9p$gender_code[df.aco.9p$gender=='male'] <- 1

sum(is.na(df.aco.9p))
df.aco.9p <- df.aco.9p%>%na.omit() # no NA

glimpse(df.aco.9p)

##
# Dataset: acoustic+articulatory w/ time info (9 timepoint)
df.aa <- read_csv('acoustic_art.csv')
sum(is.na(df.aa))
# keep only mean vals
df.aa <- df.aa[!grepl('00',colnames(df.aa))] # remove all timepoints
df.aa <- df.aa[!grepl('u',colnames(df.aa))] # remove uncorrected formants
df.aa <- df.aa[!grepl('shrF0',colnames(df.aa))] # remove shrF0
df.aa <- df.aa[!grepl('strF0',colnames(df.aa))] # remove strF0
df.aa <- df.aa[!grepl('sF0',colnames(df.aa))] # remove sF0
df.aa <- df.aa[!grepl('pF1',colnames(df.aa))] # remove praatF1
df.aa <- df.aa[!grepl('pF2',colnames(df.aa))] # remove pF2
df.aa <- df.aa[!grepl('pF3',colnames(df.aa))] # remove pF3
df.aa <- df.aa[!grepl('pF4',colnames(df.aa))] # remove pF4
df.aa <- df.aa[!grepl('pB',colnames(df.aa))] # remove pBx

sum(is.na(df.aa))

df.aa <- df.aa%>%na.omit() # no NA
unique(df.aa$Filename[df.aa$type=='fry'])#179
unique(df.aa$Filename[df.aa$type=='pd'])#415
unique(df.aa$Filename[df.aa$type=='modal'])#324

df.aa$gender_code[df.aa$gender=='female'] <- 0
df.aa$gender_code[df.aa$gender=='male'] <- 1

glimpse(df.aa)

# Section 1: set up x,y ####
df <- df.aco
df <- df.aco.9p
df <- df.aa

# assigning predictors & responses
glimpse(df)
dim(df)
y <- df$type
table(y)
x <- df[,5:37] # depend on the num of features
x <- df[,5:325] # depend on the num of features
x <- df[,5:48] # depend on the num of features

# scaling x
x_scale <- scale(x,center = T,scale = T)

# Section 2: correlation ####
cortable <- cor(x,use = 'complete.obs')
corrplot(cortable, tl.cex=0.7,tl.pos = 'lt', tl.srt = 45, tl.col = 'black')

# Section 3: t-SNE ####
# subset the categorical info & assign unique row IDs
df_meta <- df %>%
  select(c(1:4))%>%mutate(ID=row_number())

# use t-SNE to visualize data using x (predictors) onto 2-dim space
set.seed(102)
tSNE_fit <- df %>%
  select(c(5:37)) %>%
  scale() %>%
  Rtsne(initial_dims=26,perplexity=50)
# equivalent to: (preferred)
tSNE_fit <- x_scale %>% 
  Rtsne(initial_dims=26,perplexity=50)

#initial_dims: the number of dimensions that should be retained in the initial 
# PCA step (default: 50)
#perplexity should not be bigger than 3 * perplexity < nrow(X) - 1

tSNE_df <- tSNE_fit$Y %>% 
  as.data.frame() %>%
  rename(tSNE1="V1",
         tSNE2="V2") %>%
  mutate(ID=row_number())

# join back the meta information
tSNE_df <- tSNE_df %>%
  inner_join(df_meta, by="ID")

library(wesanderson)
style <- list(
  scale_color_manual(values = wes_palette("IsleofDogs1")[c(3,4,1)]),
  scale_fill_manual(values = wes_palette("IsleofDogs1")[c(3,4,1)]),
  theme_light(base_size = 24),
  theme(
    legend.position = 'bottom'
  )
)

tSNE_df %>%
  ggplot(aes(x = tSNE1, 
             y = tSNE2,
             color =type,
             shape=gender))+
  geom_point(alpha=0.8)+style#+facet_wrap(~gender)


# Section 4: umap ####
# subset the categorical info & assign unique row IDs
df_meta <- df %>%
  select(c(1:4))%>%mutate(ID=row_number())

# use umap to visualize data using x (predictors) onto 2-dim space
set.seed(102)
umap_fit <- x_scale %>% 
  umap()

umap_df <- umap_fit$layout %>% 
  as.data.frame() %>%
  rename(umap1="V1",
         umap2="V2") %>%
  mutate(ID=row_number())

# join back the meta information
umap_df <- umap_df %>%
  inner_join(df_meta, by="ID")

umap_df %>%
  ggplot(aes(x = umap1, 
             y = umap2,
             color =type))+
  geom_point(alpha=0.8)+facet_wrap(~gender)+style

# Section 5: DIY ####
# try on other dfs
# try subsetting x (predictors)
# try on different y (meta information): voice type, gender, ID
# try manipulating t-sne/umap parameters

colors = rainbow(length(unique(df$type)))
names(colors) = unique(df$type)
tsne_plot <- function(perpl=30,iterations=500,learning=200){
  set.seed(102) # for reproducibility
  tsne_fit <- Rtsne(df[,-c(1:4)], dims = 2, perplexity=perpl, verbose=TRUE, max_iter=iterations, eta=learning)
  plot(tsne_fit$Y, t='n', main = print(paste0("perplexity = ",perpl, ", max_iter = ",iterations, ", learning rate = ",learning)), xlab="tSNE1", ylab="tSNE2", "cex.main"=1, "cex.lab"=1.5)
  text(tsne_fit$Y, labels=df$type, col=colors[df$type])
}

perplexity_values <- c(2,5,30,50,100)
sapply(perplexity_values,function(i){tsne_plot(perpl=i)})

iteration_values <- c(10,50,100,1000)
sapply(iteration_values,function(i){tsne_plot(iterations=i)})

# https://cran.r-project.org/web/packages/umap/vignettes/umap.html
# https://www.analyticsvidhya.com/blog/2017/01/t-sne-implementation-r-python/

umap_plot <- function(n_neighbors=15,n_epochs=200,metric='euclidean',min_dist=.1){
  set.seed(102) # for reproducibility
  umap_fit <- umap(df[,-c(1:4)], n_components = 2, n_neighbors=n_neighbors, n_epochs=n_epochs, metric=metric,min_dist=min_dist)
  plot(umap_fit$layout, t='n', main = print(paste0("n_neighbors = ",n_neighbors, ", n_epochs = ",n_epochs, ", metric = ",metric)), xlab="umap1", ylab="umap2", "cex.main"=1, "cex.lab"=1.5)
  text(umap_fit$layout, labels=df$type, col=colors[df$type])
}

n_neighbors_values <- c(2, 5, 10, 20, 50, 100, 200)
sapply(n_neighbors_values,function(i){umap_plot(n_neighbors=i)})

n_epochs_values <- c(50,100,200,500,1000)
sapply(n_epochs_values,function(i){umap_plot(n_epochs=i)})

min_dist_values <- c(0.0, 0.1, 0.25, 0.5, 0.8, 0.9)
sapply(min_dist_values,function(i){umap_plot(min_dist=i)})

metric_values <- c("euclidean","manhattan")
sapply(metric_values,function(i){umap_plot(metric=i)})
