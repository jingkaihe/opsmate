# Get the currently used golang install path (in GOPATH/bin, unless GOBIN is set)
ifeq (,$(shell go env GOBIN))
GOBIN=$(shell go env GOPATH)/bin
else
GOBIN=$(shell go env GOBIN)
endif

VERSION=$(shell awk '/^\[project\]/{p=1;next} /^\[/{p=0} p&&/^version = /{print}' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
IMAGE_NAME=opsmate
CONTAINER_REGISTRY=ghcr.io/jingkaihe

SHELL = /usr/bin/env bash -o pipefail
.SHELLFLAGS = -ec

LOCALBIN ?= $(shell pwd)/.bin

KIND ?= $(LOCALBIN)/kind

## Location to install dependencies to
LOCALBIN ?= $(shell pwd)/bin
$(LOCALBIN):
	mkdir -p $(LOCALBIN)

docker-build:
	docker build -t $(CONTAINER_REGISTRY)/$(IMAGE_NAME):$(VERSION) .

docker-push:
	docker push $(CONTAINER_REGISTRY)/$(IMAGE_NAME):$(VERSION)
	docker tag $(CONTAINER_REGISTRY)/$(IMAGE_NAME):$(VERSION) $(CONTAINER_REGISTRY)/$(IMAGE_NAME):latest
	docker push $(CONTAINER_REGISTRY)/$(IMAGE_NAME):latest

.PHONY: kind
kind: $(LOCALBIN)
	test -s $(LOCALBIN)/kind || curl -Lo $(LOCALBIN)/kind https://kind.sigs.k8s.io/dl/v0.24.0/kind-linux-amd64 && chmod +x $(LOCALBIN)/kind

.PHONY: kind-cluster
kind-cluster: kind
	$(KIND) create cluster --config eval/bootstrap/kind.yaml

.PHONY: kind-destroy
kind-destroy: kind
	$(KIND) delete cluster --name troubleshooting-eval

.PHONY: api-gen
api-gen: # generate the api spec
	echo "Generating the api spec..."
	poetry run python scripts/api-gen.py

.PHONY: python-sdk-codegen
python-sdk-codegen: api-gen # generate the python sdk
	echo "Generating the python sdk..."
	sudo rm -rf sdk/python
	mkdir -p sdk/python
	cp .openapi-generator-ignore sdk/python/.openapi-generator-ignore
	docker run --rm \
		-v $(PWD)/sdk:/local/sdk \
		openapitools/openapi-generator-cli:v7.10.0 generate \
		-i /local/sdk/spec/apiserver/openapi.json \
		--api-package api \
		--model-package models \
		-g python \
		--package-name opsmatesdk \
		-o /local/sdk/python \
		--additional-properties=packageVersion=$(VERSION)
	sudo chown -R $(USER):$(USER) sdk

.PHONY: go-sdk-codegen
go-sdk-codegen: # generate the go sdk
	echo "Generating the go sdk..."
	sudo rm -rf cli/sdk
	mkdir -p cli/sdk
	cp .openapi-generator-ignore cli/sdk/.openapi-generator-ignore
	docker run --rm \
		-v $(PWD)/cli/sdk:/local/cli/sdk \
		-v $(PWD)/sdk/spec/apiserver/openapi.json:/local/openapi.json \
		openapitools/openapi-generator-cli:v7.10.0 generate \
		-i /local/openapi.json \
		--api-package api \
		--model-package models \
		-g go \
		--package-name opsmatesdk \
		--git-user-id jingkaihe \
		--git-repo-id opsmate/cli/sdk \
		-o /local/cli/sdk \
		--additional-properties=packageVersion=$(VERSION),withGoMod=false
	sudo chown -R $(USER):$(USER) cli/sdk
