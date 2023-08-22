Anaconda3: https://www.anaconda.com/download/

Create a python virtual enviroment with anaconda

user@xyz:~/git-repos$ conda create -n yourenvname python=3.10 anaconda

Activate the enviroment

user@xyz:~/git-repos$ conda activate yourenvname

Clone the repository and navigate inside

(yourenvname) user@xyz:~/git-repos$ https://github.com/sappyb/llmmemory.git
(yourenvname) user@xyz:~/git-repos$ cd ./llmmemory
(yourenvname) user@xyz:~/git-repos/llmmemory$ 

Run setup

pip install -r requirements.txt -e .

Run

streamlit run app.py
