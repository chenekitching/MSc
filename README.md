This GitHub repository contains the relevant scripts used for my thesis, titled "Development of an open-source variant prioritisation tool".

## ML
This folder contains the following scripts:
| Script | Description | 
|-----------|---------------|
| binary_nested_cv.R | The script used to perform nested cross validation on the binary dataset. |
| multinom_nested_cv.R | The used to perform nested cross validation on the multinomial dataset. |

And the following files:
| Script | Description | 
|-----------|---------------|
| binary_ml_dataset_04-02.csv | The binary dataset consisting of P/LP and B/LB variants. |
| multinom_ml_dataset_04-02.csv | The multinomial dataset consisting of P/LP and B/LB variants |

## Singularity
This folder contains definition files to build the following containers:
| Definition file | Software | 
|-----------|---------------|
| bcftools_cont.def | BCFtools   | 
| r_cont.def | R, with the following packages: dplyr, readr, tidymodels, modelr, ranger | 
| shiny_cont.def  |  R, with the following packages: shiny, ggplot2, shinydashboard, readr, reactable, dplyr, DT, flexdashboard, fresh, shinyBS, conflicted | 
