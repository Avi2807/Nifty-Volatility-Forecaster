PY ?= python

features: ; $(PY) -m scripts.build_features
sentiment: ; $(PY) -m scripts.build_sentiment
baselines: ; $(PY) -m scripts.run_baselines
nn: ; $(PY) -m scripts.train_nn --model mlp
eval: ; $(PY) -m scripts.run_all_eval
test: ; $(PY) -m tests.run_tests
