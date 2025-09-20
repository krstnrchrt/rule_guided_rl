# im_env_private
Private github repo

This project can be split into two sections.


1) Text Simplification
     - make sure to set up a conda environment using the file im_env.yaml
     - make sure to download the spacy language model as follows: <python -m spacy download de_core_news_lg>
     - install german_compound_splitter (https://github.com/repodiac/german_compound_splitter?tab=readme-ov-file)
            - follow installation steps as described above
            - download the dictionary (https://sourceforge.net/projects/germandict/files/latest/download) and put it in the resources/german_dict folder structure

2) Reinforcement Learning
     - set up the conda environment using the file trl-env.yaml
     - make sure to download the spacy language model as follows: <python -m spacy download de_core_news_lg>
     - install german_compound_splitter (https://github.com/repodiac/german_compound_splitter?tab=readme-ov-file)
            - follow installation steps as described above
            - download the dictionary (https://sourceforge.net/projects/germandict/files/latest/download) and put it in the german_dict/ folder
