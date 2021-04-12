# Deep Onco AI

This analysis pipeline is meant to allow the training of multiple machine-learning models 
on various subsets of the data in the large cell line repositories like the CCLE and GDSC. 
Overall predictors can be designed by ensembling the most successful models for each drug. 



## Description

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

## Getting Started

### Dependencies

* Python3
* To install all necessary modules use `pip3 install -r requirements.txt`
* Some data must be downloaded externally to the */data* folder:
	CCLE_RNAseq_genes_rpkm_20180929

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

Furthermore, all filters are fitted on the original data and then applied 
in no particular order, In other words, the only features that are retained 
are the ones that pass all the filters.

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

The same filters are applicable to the 'omics' and 'targets'. Target normalization is only used 
when the overall normalization is deactivated (in *master_script.py*). For target engineering, 
only the quantization method is currently implemented. Thresholding will be active in a future release.

The different omics or targets can be commented out of the analysis. Within each omic or target, 
the different filters and other methods can be disabled individually (enabled: false)

In the *modeling* part, the options used for the analysis can be specified, for example 
the number of folds for the different cross-validations, the random seeds, the search 
depth of the hyperparameter optimization step, the metric used for performance and the 
configuration of the ensembling step.

#### Step-by-step

* modify the file *config.yaml*
* run `python3 master_script.py`
* the results of the stacking is written as a *pickle* object
* the analysis is recored in *run.log*

### Functions


## Help

Contact the authors for any help in using the tools.

## Contributing

Contributions are welcome from mebers of the group. Look for the TODO keyword. Here is 
a brief list of things yet to implement:

* upsampling with SMOTE and VAEs
* thresholding of responses in combination with the quantization
* loading of the 'BinarizedIC50' values (alternative targets)
* more models to hyper-optimize (elastic net, NN architectures)
* stacking with more algorithms
* stack of stacks
* load mutational, metabolomic data
* compile gene-level versus transcript level expression
* more filters (outliers, ...)
* grid-search to compare with bayesian search
* add dunder methods for classes
* 


## Authors

Sébastien De Landtsheer
sebdelandtsheer@gmail.com
DeepBioModeling
contact@deepbiomodeling.com
https://www.deepbiomodeling.com
Prof. Thomas Sauter
thomas.sauter@uni.lu

## Version History

* 0.1
    * Initial Release April 2021

## License

?

## Acknowledgments