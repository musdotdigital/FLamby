.ONESHELL:

CONDAPATH = $$(conda info --base)

# set default value
enable ?= all_extra
install:
	conda env create -f environment.yml
	${CONDAPATH}/envs/fed-substra/bin/pip install -r requirements.txt

install-mac:
	conda env create -f environment.yml
	conda install nomkl
	${CONDAPATH}/envs/fed-substra/bin/pip install -r requirements.txt

update:
	conda env update --prune -f environment.yml
	${CONDAPATH}/envs/fed-substra/bin/pip install -U requirements.txt

clean:
	conda env remove --name fed-substra