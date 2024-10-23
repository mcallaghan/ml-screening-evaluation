## The safe use of LLMs for screening in systematic reviews

### Background


The field of Machine Learning (ML) for evidence synthesis aims to develop applications of machine learning that save labour by supplementing or replacing human effort at various stages of the systematic review process. Since ChatGPT popularised Large Language Models (LLMs) in November 2022, several projects have assessed the potential to use LLMs to increase the labour savings possible when applying ML technologies to systematic reviews. Some of these projects have looked at the task of screening, where a mature literature exists on how to manage the uncertainty that comes with any application of ML in order to ensure that the quality of reviews is not compromised. However, the hitherto available evaluations of LLMs for screening have not sufficiently engaged with this existing literature, such that we do not yet know the extent to which they may reduce labour in a realistic setting where the risk of missing relevant studies can be appropriately managed.

### Objectives

- To outline a framework for evaluating LLMs for screening in a way compatible with their safe use in real projects, by combining with stopping criteria.
- To assess the extent to which using LLMs for screening may offer additional labour savings as compared to standard methods, while maintaining high standards
- To assess the extent to which model choice and prompting strategy affect potential labour savings
- To quantify the additional costs involved in using LLMs for screening instead of standard methods

### Results

Current results show that, using simple prompts based solely on the review title, LLMs result in substantially smaller labour savings than standard machine-learning prioritisation pipelines using support vector machine classifiers. Further results will show the effect of more complex prompting strategies involving study inclusion criteria, as well as the comparative costs of LLM-assisted screening and traditional approaches.

### Conclusions

Though LLMs offer impressive capabilities given their zero-shot nature, their simplistic application in systematic review screening may result in smaller work savings than current approaches can deliver. A future research agenda may find ways to combine active learning approaches with LLMs, for example by automatically adjusting prompts based on inclusion and exclusion decisions.
