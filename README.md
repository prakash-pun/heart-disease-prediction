# cardiovascular-disease-prediction

A group project on cardiovascular disease prediction project

## Setup Initial Project

```bash
git clone https://github.com/OpenMinds-LCIT/cardiovascular-disease-prediction.git
```

## Working on Branch

- When working on new task/feature, create a new branch then work on that branch. format of the branch `git branch <your_name-task/feature>`
  - e.g. `git checkout -b your-name/data-cleaning-age`
- Developer always have to pull `git pull` code from `main` branch and sync with it.
- Make a PR(Pull Request) after completion of task/feature and always add a code reviewer in associated PR.
- Make a habit to check on your PR, so that you can follow along with status, comments, tasks of your PR.
- Once your PR is approved and merged to main branch then you can switch back to main branch and ready to work on another feature/task.

```bash
git branch # to list branch
git checkout -b <branch_name> # create new branch and move to new branch
git checkout <branch_name> # to change branch

git add . # to add your changes
git commit -m "your commit message" # to commit your changes
git status # to check your status
git log # to checkout your commit logs
git push # to push your code to github
git pull # to pull new changes from github
```

## Folder Structure

```bash
|____ data # Data sets e.g excel files, csv
|____ docs # Add your project report, task report e.g. word files
|____ notebooks # Jupter Notebooks
|____ slides and charts # General analysis as charts in HTML/PDF/PPTS/IMAGES
|____ src # Source code that is used in the project
|____ .gitignore # ignore dir
|____ requirements.txt # all the library used in the project
|____ run.sh # bash script to run main.py
|____ STANDARD.md # pep8 docs
|____ stream.sh # bash script to run stream lit dashboard
```

## Work on Virtualenv

```bash
> pip install virtualenv
> python -m venv venv # for windows
> python3 -m venv venv # for mac
> venv/Scripts/activate # to activate virtualenv windows
> source venv/bin/activate # to active virutalenv mac
> pip install -r requirements.txt # to install all the packages
> pip freeze > requirements.txt # to update requirements.txt
```

## Run the project

```bash
bash run.sh # run the project (main.py file)
bash stream.sh # run the dashboard (dashboard.py file)

or

python src/main.py
streamlit run src/dashboard.py
```

### Happy Coding :smiley:
