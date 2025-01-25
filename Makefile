# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This Makefile includes all targets related to CI/CD tests for NVIDIA's MLPerf Inference codebase


# If specific DOCKER_COMMAND is not passed, launch interactive docker container session.
ifeq ($(DOCKER_COMMAND),)
    DOCKER_INTERACTIVE_FLAGS = -it
else
    DOCKER_INTERACTIVE_FLAGS =
endif

ifeq ($(DOCKER_GPU),)
    DOCKER_GPU_FLAGS =
else
    DOCKER_GPU_FLAGS = --gpus=$(DOCKER_GPU) -e NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES}
endif
DOCKER_REPO ?= gitlab-master.nvidia.com/mlpinf/mitten
DOCKER_TAG ?= main-latest
DOCKER_GIT_TAG := $(shell git rev-parse --short HEAD 2> /dev/null)
DOCKER_DIST ?= x86_64
TEST_FLAGS := -s --durations 0 -vv -rXfw


.PHONY: build_docker
build_docker:
	docker build -t mitten:$(DOCKER_TAG) \
		--network host \
		-f docker/Dockerfile.$(DOCKER_DIST) .


.PHONY: launch_docker
launch_docker: build_docker
	docker run --rm \
		$(DOCKER_INTERACTIVE_FLAGS) \
		$(DOCKER_GPU_FLAGS) \
		-w /opt/mitten \
		--cap-add SYS_ADMIN --cap-add SYS_TIME \
		--shm-size=32gb \
		-v /etc/timezone:/etc/timezone:ro \
		-v /etc/localtime:/etc/localtime:ro \
		-v ${CURDIR}:/opt/mitten-develop \
		--security-opt apparmor=unconfined \
		--security-opt seccomp=unconfined \
		--runtime=nvidia \
		mitten:$(DOCKER_TAG) $(DOCKER_COMMAND)


.PHONY: push_docker
push_docker:
	docker tag mitten:$(DOCKER_TAG) $(DOCKER_REPO):$(DOCKER_GIT_TAG)
	docker push $(DOCKER_REPO):$(DOCKER_GIT_TAG)


.PHONY: build
build:
	python3 setup.py build


.PHONY: install
install: build
	python3 -m pip install .


.PHONY: install_test_deps
install_test_deps:
	python3 -m pip install .[test]


.PHONY: unit_tests
unit_tests: install_test_deps
	python3 -m pytest $(TEST_FLAGS)


.PHONY: pep8
pep8:
	python3 -m pycodestyle --max-line-length=120 src/


.PHONY: pylint
pylint:
	python3 -m pylint src/


.PHONY: dist
dist:
	rm -rf dist/
	python3 -m pip install build
	python3 -m build
	cp dist/*.whl ../mitten-develop/dist
