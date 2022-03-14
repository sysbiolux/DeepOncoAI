# Deep Onco AI

This analysis pipeline is meant to allow the training of multiple machine-learning models
on various subsets of the data in the large cell line repositories like the CCLE and GDSC.
Overall predictors can be designed by ensembling the most successful models for each drug.

The goal of the current pipeline is to : 
* evaluate the predictivity of the different algorithms on the different subsets of the data, on the different drugs.
* retrieve and compare the most important features for these, for each drug
  * do similar algorithms pick up similar signals?
  * which algorithms are better at which task?
  * are the observations conserved across drugs?
* build ensemble predictors:
  * are ensembles of 2 predictors more performant than single algorithms? Which ones and in which case?
  * what are the 'best' ensembles for each drug, can a biomarker signature of resistance/sensitivity be formed?

## Description and rationale

The prediction of individual patients' response to chemotherapy is a central problem in oncology.
Cell lines can resume some of the characteristics of the patients original tumors and
can be screened with various drugs. By using the baseline gene and protein expression
of the cells it is possible to build predictors of response for cell lines and utlimately
patients before treatment or after relapse.
The Systems Biology group of the DLSM led by Prof. Thomas Sauter has extensive expertise
in the fields of network biology and computational modeling, with applications in
drug repurposing and target identification, among others.
DeepBioModeling is a young startup with a focus on custom machine-learning solutions for biological
data analysis and a dedication to professional software engineering.

## Towards a pipeline

The pipeline was separated into primary logical chunks that can be run independently.

Here are aexamples of run configurations
* load: --overwrite -c input\config.yaml -o testdir -f testdir\raw.pkl

* filter: --overwrite -c input\config.yaml -i testdir\raw.pkl -o testdir -f testdir\filtered.pkl

* preprocess: --overwrite -c input\config.yaml -i testdir\filtered.pkl -r testdir\raw.pkl -o testdir -f testdir\preprocessed.pkl

* model: --overwrite -c input/config.yaml -i testdir\preprocessed.pkl -o testdir -f testdir\trained_models.pkl

* modeling_results: --overwrite -c input\config.yaml -i testdir\trained_models.pkl -o testdir

* stacks: tbc

## Getting Started

### Dependencies

* Python3
* Conda
* Snakemake [Snakemake getting started](https://snakemake.readthedocs.io/en/stable/getting_started/installation.html)
* All necessary modules will be automatically installed during the execution of the snakemake pipeline. Actually 
  each 'chunk' has its own set of dependencies and environment.
* Some data must be downloaded externally to the */data* folder:
  CCLE_RNAseq_genes_rpkm_20180929 can be found on the CCLE website

### Use

#### General

The pipeline runs by itself given the info in the *config.yaml* config file.
Here is some info about [yaml](https://yaml.org/)
The configuration is organized as follows:

- data:
  - omics:
    - omic1
    - omic2
    - ...
  - targets:
    - target1
    - target2
    - ...
- modeling:
  - options...

The *data* part lists the different subsets used for analysis.
The pipeline will recognize which omic types are needed from which database and will
import the necessary files.

Under each omic, the following info is allowed:

- omic:
  - name:
  - database:
  - filtering:
    - filter1
    - filter2
    - ...
  - feature_engineering:
    - feature_selection
      - selection1
      - selection2
    - transformations
      - transformation1
      - transformation2

The following filters are available:

* sample_completeness: removes samples with insufficient data
* feature_completeness: removes features with insufficient data
* feature_variance: removes features with insufficient variance
* cross-correlation: removes cross-correlated features (experimental optimization
  for large datasets instead of exact solution)

Furthermore, filters are divided into 'fast' and 'slow'. Fast filters are fitted and applied first to reduce 
the size of the dataset. Slow filters (cross-correlation) are fitted on the result of the first pass and applied 
subsequently. Filters are additive, i.e. the only features that are retained
are the ones that pass all the filters in each of the two filtering steps.

The following selections are available:

* importance: selects the top features according to XGBoost
* predictivity: selects the top features by cross-elimination (not recommended
  for large datasets)

These methods will select the features with the most signal for the next step.

The following transformations are available:

* PCA
* t-SNE
* Polynomial combination
* 'OR gate' combination
  Future updates will include ICA and RPA. The selected features are transformed or
  combined into a new dataset of features.

Under each target, the following info is allowed:

- target:
  - name
  - database
  - responses (which metric is used for the response)
  - target_drug_name
  - filtering:
    - filter1
    - filter2
    - ...
  - normalization
  - target_engineering:
    - method1
    - method2

The same filter types are applicable to the 'omics' and 'targets'. Target normalization is only used
when the overall normalization is deactivated (in *master_script.py*). For target engineering,
only the quantization method is currently implemented. Thresholding will be active in a future release.

The different omics or targets can be commented out of the analysis. Within each omic or target,
the different filters and other methods can be disabled individually (enabled: false)

In the *modeling* part, the options used for the analysis can be specified, for example
the number of folds for the different cross-validations, the random seeds, the search
depth of the hyperparameter optimization step, the metric used for performance and the
configuration of the ensembling step.

#### Step-by-step

* create a new environment:
```
$ conda install -n base -c conda-forge mamba
$ conda activate base
$ mamba create -c conda-forge -c bioconda -n snakemake snakemake

$ conda activate snakemake
```
* copy the file *config.yaml* in a new folder and modify it as needed
* specify this folder name in `config_snake.yaml` as the input, and a path for the output
* run the pipeline from the top-level folder:
```
snakemake --cores 1 --use-conda --configfile workflow\config_snake.yaml
```
(remember the slash is inverted in Unix systems)

* the results of each of the steps is written as a *pickle* object. Here is some info
  about [pickle](https://docs.python.org/3/library/pickle.html)
* the analysis is recorded in the snakemake run log.

### Structure

Alright, you want to dig in the code. Here is some useful info: 

* chunks: `load.py`, `filter.py`, `preprocess.py`, `model.py`, `stack.py`
* the highest-level functions are located in `config.py`
* data is organized with samples as lines and features as columns
* the `Dataset` class is used throughout the project. It contains a Pandas Dataframe, and two
  Pandas Series corresponding to the 'omic' and 'database' of each feature (columns)
* there are two `Filter` classes: one for samples, and one for features. That is because the
  samples flavor is applied once, whereas internal validation would require that preprocessing
  is applied to both training and test subsets but trained only on the training set. As the
  number of samples might be too low at this point of the project this is not implemented yet but
  the filtering concept is already in place. The features flavor of filters need an instance of
  the `Rule` class, whereas the samples flavor does not.
* filters are separated into fast (applied first) and slow (applied second) to decrease computation time.
* the pipeline can be run either in 'optimization' mode, where hyperparameters of each classifier is
  performed with Bayesian optimization, or in 'standard' mode where a predefined set of hyperparameters
  is used for all trainings. At this point small increases in predictivity have been observed using hyperoptimized
  models, and minimum differences observed between hyper-optimized parameter values and default ones, but this 
  has not been investigated in full.
* ...

### Visualizations:

Both data and results are visualized. Here is the list of all available plots for each 'omic':

* general distribution of the features before and after log transformation. If more than 100 features are present
  a sample of 100 is created at random
* mean vs variance scatterplot, before and after log transformation
* analysis of missing data: fraction of data present per sample, per feature, and
  binary map of missing data
* analysis of missing data correlation:  per sample, per feature: histograms of
  correlation coefficients and heatmaps of cross-correlations of missing data presence
* correlation analysis: histograms of correlation coefficients and heatmap of data cross-correlations, for
  both samples and features
* target analysis: distributions of raw values, and visualization of the thresholds on the
  distribution of normalized values

## Help

Contact the authors for any help in using the tools.

## Contributing

Contributions are welcome from members of the group. Look for the TODO keyword. Here is
a brief list of things yet to implement:

* upsampling with SMOTE and VAEs
* thresholding of responses in combination with the quantization
* loading of the 'BinarizedIC50' values (alternative targets)
* more models to hyper-optimize (elastic net, NN architectures)
* stacking with more algorithms for scikit-learn or others
* stack of stacks
* load mutational data
* compile gene-level versus transcript level expression
* more filters (outliers, ...)
* grid-search or other to compare with bayesian search
* add dunder methods for classes
* unit tests
* include GDSC data


## Authors

- Sébastien De Landtsheer sebdelandtsheer@gmail.com DeepBioModeling
  contact@deepbiomodeling.com https://www.deepbiomodeling.com
- Prof. Thomas Sauter thomas.sauter@uni.lu


## Version History

* 0.1
  * Initial Release April 2021

## License

?

## Acknowledgments
