# Rule-Guided RL for German Text Simplification

A research codebase exploring rule-guided German text simplification: first via a deterministic rule pipeline, then by transforming those rules into a numeric reward to fine-tune LLMs with RLHF (PPO).

This project is split into 2 sections.

# Environments & Setup

1) Text Simplification
     - set up a conda environment using the file im_env.yaml
     - download the spacy language model as follows: python -m spacy download de_core_news_lg
     - install german_compound_splitter (https://github.com/repodiac/german_compound_splitter?tab=readme-ov-file)
     - follow installation steps as described above
     - download the dictionary (https://sourceforge.net/projects/germandict/files/latest/download) and put it in the _resources/german_dict_ folder
     - download the verbs.csv file from here (https://github.com/viorelsfetea/german-verbs-database/tree/master/output) and put it under resources/

2) Reinforcement Learning
     - set up the conda environment using the file trl-env.yaml
     - make sure to download the spacy language model as follows: <python -m spacy download de_core_news_lg>
     - german compound splitter: follow the same steps as above. Place the dictionary under trl_directory/german_dict


# Workflow Description

## 1. Text Simplification

1. Prepare and Simplify 
- Place source files into master_data/0_original/ 
- run [original_file_sentence_extraction.ipynb](original_file_sentence_extraction.ipynb) to create the base sentence dataset
- run [preprocess_preparse.ipynb](preprocess_preparse.ipynb) to apply preprocessing and parsing
- run [text_simplification.ipynb](text_simplification.ipynb) (rule-based pipeline which outputs:)
     - simplified text -> [master_data/3_simplified](master_data/3_simplified)
     - per-sentence logs -> [simplification_logs/](simplification_logs)

2. Aggregate Logs & Compute Metrics
- run [assess_rule_output.ipynb](assess_rule_output.ipynb) to aggregate logs for downstream steps
- ensure the variables point to the files in simplification_logs/
- output:
     - aggregated, cleaned file of simplifications saved in -> [master_data/output_assessment](master_data/output_assessment)
     - calculated bertscore saved in -> [master_data/output_assessment](master_data/output_assessment)
- run [assess_bertscore.ipynb](assess_bertscore.ipynb) for a more readable summary

## 2. Reinforcement Learning

1. Data Placement
- copy the simplified, cleaned dataset (saved in [master_data/output_assessment](master_data/output_assessment)) into [trl_directory/data](trl_directory/data)
- the dataset including split used within this project can be downloaded from GoogleDrive [here](https://drive.google.com/drive/folders/12tqvIS3Y1oTr9QVL0KqLNHITN14lgTem?usp=sharing)
     - save it in [trl_directory/sft_split_dataset](trl_directory/sft_split_dataset)

2. Reward Model (RM) Training
- run [rm_training.ipynb](trl_directory/rm_training.ipynb)
- outputs a trained reward model

3. SFT Training
- in step 2. the data from [trl_directory/data/] is accessed, cleaned, processed and formated
     - if you download the split data from the drive link, then uncomment step __2.__ and begin from step __2.5__
     - if you follow the notebook's flow, then UNCOMMENT step __2.5__
- run [trl_directory/sft_training.ipynb](trl_directory/≈)


3. PPO Training
- several variables need to be set according to which experiment is run
     - 1) Are you implementing a direct PPO training or with an SFT step in-between?
          - __POLICY_MODEL_ID__ directly pluck in model ID
          - __SFT_MODEL_OUTPUT_DIR__ pluck in name of sft_model trained in step 3
     - 2) Set the query format to simple query (Experiment 1A) or more detailed query (Experiment 1B, 2A, 2B)
     - 3) Adjust the trained PPO file name
     - 4) Set the training parameters
     - 5) Make sure the split data is within the correct directory      
- run [ppo_training.ipyn](trl_directory/ppo_training.ipynb), pluck in trained RM here during training process

4. Evaluation
- Make sure the corresponding eval.csv data is within expected directory
- Adjust the query format as explained in 3.2
- Adjust the output file name and the trained PPO_model name that needs to be evaluated 
- Run [ppo_evaluation.ipynb](trl_directory/ppo_evaluation.ipynb)
- The notebook [final_eval_calc.ipynb](trl_directory/final_eval_calc.ipynb) performs final metric calculation and a deep dive into performance