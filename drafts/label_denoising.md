Label denoising

Sources of label noise:
	Poorly framed task
		Evolution of task and understanding of what labels are
		Different understanding of edge cases (not well defined, subjective)
	Labelling mistakes
		Fatigue
	Bad faith actors
	Innapropriate features (not enough info to find the pattern)
		
Label noise taxonomy
	Feature dependent
		Independence of features (sometimes correlations will be incidental)
		
	Uncertainty
		aLEATORIC: Label noise that cannot be corrected with more data (e.g., broken sensor, labeller misunderstanding)
			Homoscedastic: Evenly distributed label noise
			Heteroscedastic: Area specific label noise 
		Epistemic Uncertainty:
			Label noise that can be corrected with more data
	
Noisy label resolution techniques:
	1.  Hard filtering: O2U-Net, co-teaching
	2.  Soft filtering: detecting and reweighting noisy isntances
	3. Correction: correct noisy instances (Meta Label Correction)

O2U-Net
	Hard labels are memorized at end of training
	Look at indivual losses per item and remove items that have highless early on and low losses later on
	O2U-NET looks at cyclical nature
	
Bayesian Active Learning by Disagreement (BALD, BatchBald)
	?
Meta label correction (MLC)
	Train a golden reference model

Co-teaching
	Have multiple (two) networks learning. Peer review and look based on mismatches (need to read)
	

	
https://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_O2U-Net_A_Simple_Noisy_Label_Detection_Approach_for_Deep_Neural_ICCV_2019_paper.pdf
https://arxiv.org/abs/1911.03809
https://arxiv.org/abs/1711.00583
https://arxiv.org/abs/1804.06872
https://arxiv.org/abs/1906.08158

General takeaway:
	Need to store labelling context and information (need to have a trace)
	
	
Uncertainty team

Random thought:
	What if the inability to learn an example is due to 