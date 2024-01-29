PHONY: tiktoken

tiktoken:
	# Install in current env
	. .venv/bin/activate;                           \
	cd nanogpt_deepdive/data/shakespeare_tiktoken/; \
	pip install -e .
