WIP readme

# Setup


1. Create an environment from environment.yml
` conda env create -f environment.yml `
2. Activate the environment
` conda activate wsc `
3. Download some spacy stuff
` python -m spacy download en_core_web_lg ` 
4. WSC data can be downloaded [here](https://super.gluebenchmark.com/tasks), Winogrande data can be downloaded [here](https://mosaic.allenai.org/projects/winogrande) 


# About MLM

{"text": "Jane gave Jasmine candy because she wasn't hungry.", "target": {"span2_index": 5, "span1_index": 2, "span1_text": "Joan", "span2_text": "she"}, "idx": 21, "label": false}


Raw input
Jane gave Jasmine candy because she wasn't hungry.

MC_MLM

Jane gave Jasmine candy because \[MASK\] wasn't hungry.
Jane gave Jasmine candy because Jane wasn't hungry.
log p( Jane | MASK )

Jane gave Jasmine candy because \[MASK\] \[MASK\] \[MASK\] wasn't hungry.
Jane gave Jasmine candy because \[MASK\] \[MASK\] \[MASK\] wasn't hungry.
mean (  log p(J | MASK1), log p(as | MASK2), log p(mine | MASK3))


# About multiple candidates
Jane gave Jasmine and Joan candy because she wasn't hungry.  (2 candiates)

Jane gave Joan candy because she was hungry. (1 candidate)

[[cand1,  cand2],  [cand1, ALL_PADDING]]