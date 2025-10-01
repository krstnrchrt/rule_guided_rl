# Rule-Guided RL for German Text Simplification


A research codebase exploring rule-guided German text simplification: first via a deterministic rule pipeline, then by transforming those rules into a numeric reward to fine-tune LLMs with RLHF (PPO).


This project is split into 2 sections.


# Environments & Setup


1) Text Simplification
    - set up a conda environment using the file `im_env.yaml`
    - download the spacy language model as follows: `python -m spacy download de_core_news_lg`
    - install [german_compound_splitter](https://github.com/repodiac/german_compound_splitter?tab=readme-ov-file) and follow installation steps as described in the repository
        - download the [dictionary](https://sourceforge.net/projects/germandict/files/latest/download) and put it in the `resources/german_dict_ folder`
    - download the verbs.csv file from [here](https://github.com/viorelsfetea/german-verbs-database/tree/master/output) and put it under `resources/`


2) Reinforcement Learning
    - set up the conda environment using the file `trl-env.yaml`
    - make sure to download the spacy language model as follows: `python -m spacy download de_core_news_lg`
    - __german compound splitter__: follow the same steps as above.
       - Place the dictionary under `trl_directory/german_dict`




# Workflow Description


## 1. Text Simplification


1. Prepare and Simplify
- Place source files into `master_data/0_original/`
- run [original_file_sentence_extraction.ipynb](original_file_sentence_extraction.ipynb) to create the base sentence dataset
- run [preprocess_preparse.ipynb](preprocess_preparse.ipynb) to apply preprocessing and parsing
- run [text_simplification.ipynb](text_simplification.ipynb) (rule-based pipeline which outputs:)
    - simplified text -> `master_data/3_simplified`
    - per-sentence logs -> `simplification_logs/`


2. Aggregate Logs & Compute Metrics
- run [assess_rule_output.ipynb](assess_rule_output.ipynb) to aggregate logs for downstream steps
- ensure the variables point to the files in `simplification_logs/`
- output:
    - aggregated, cleaned file of simplifications saved in -> `master_data/output_assessment`
    - calculated bertscore saved in -> `master_data/output_assessment` for a more readable summary


## 2. Reinforcement Learning


### 1. Data Placement
- copy the simplified, cleaned dataset (saved in `master_data/output_assessment`) into `trl_directory/data`
- the dataset including split used within this project can be downloaded from GoogleDrive [here](https://drive.google.com/drive/folders/12tqvIS3Y1oTr9QVL0KqLNHITN14lgTem?usp=sharing)
    - save it in `trl_directory/sft_split_dataset`


### 2. Reward Model (RM) Training
- run [rm_training.ipynb](trl_directory/rm_training.ipynb)
- outputs a trained reward model


### 3. SFT Training
- in step 2. the data from `trl_directory/data/` is accessed, cleaned, processed and formated
- within the notebook `sft_training.ipynb`:
    - if you download the split data from the drive link, then uncomment step __2.__ and begin from step __2.5__
    - if you follow the notebook's flow, then skip step __2.5__
- run [sft_training.ipynb](trl_directory/)






### 4. PPO Model Training (`ppo_training.ipynb`)


This notebook fine-tunes the policy model using PPO. Before running, configure the variables below to replicate a specific experiment from the paper.


#### Configuration


1. **Set Input Policy Model**:
   -  Use `POLICY_MODEL_ID` for **Experiments 1A & 1B** (direct PPO).
   -  Use `SFT_MODEL_OUTPUT_DIR` for **Experiments 2A & 2B** (SFT-enhanced PPO).


2. **Configure Instruction Prompt**:
   -  In the `format_query` function, uncomment the appropriate line for the simple or detailed prompt.


3. **Define Output Model Name**:
   -  Set a descriptive name in `MODEL_FILE_NAME` (e.g., `ppo_model_exp_1A`).


4. **Set Training Parameters & Paths**:
   -  Adjust `DATASET_SIZE` and `EPOCH`.
   -  Verify the `MASTER_..._DATASET` file paths are correct.


#### Experiment Configurations


Use this table as a quick reference to set up each experiment.


| Experiment | Policy Model (Input)            | Instruction Prompt      | Target KL |
| :--------- | :------------------------------ | :---------------------- | :-------- |
| **1A** | Base Model (`POLICY_MODEL_ID`)    | Simple (`Vereinfache...`) | 0.05      |
| **1B** | Base Model (`POLICY_MODEL_ID`)    | Detailed (`Aufgabe: ...`) | 0.05      |
| **2A** | SFT Model (`SFT_MODEL_OUTPUT_DIR`) | Detailed (`Aufgabe: ...`) | 0.05      |
| **2B** | SFT Model (`SFT_MODEL_OUTPUT_DIR`) | Detailed (`Aufgabe: ...`) | 0.2       |



#### Execution
Execute the notebook cells. You will be prompted for the path to your trained **Reward Model (RM)** during the PPO training.




### 4. Evaluation
- Make sure the corresponding `eval.csv` data is within expected directory
- Adjust the query format depending on Experiment format (see Experiment Configuration)
- Det a descriptive file name (e.g. `ppo_model_exp_1A_eval.csv`) and the trained PPO_model name that needs to be evaluated
- Run [ppo_evaluation.ipynb](trl_directory/ppo_evaluation.ipynb)
     - outputs: file with simplifications and their corresponding reward value (calculated by trained RM)
     
- The notebook [final_eval_calc.ipynb](trl_directory/final_eval_calc.ipynb) performs final metric calculation and a deep dive into performance (applied on output from `ppo_evaluation.ipynb` )

