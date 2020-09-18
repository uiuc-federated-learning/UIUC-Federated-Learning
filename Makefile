PROTO = federated.proto
COORGEN = coordinator/generated
HOSPGEN = hospital/generated

.PHONY: proto

proto: $(PROTO)
	python3 -m grpc_tools.protoc -I. --python_out=$(COORGEN) --grpc_python_out=$(COORGEN) federated.proto
	python3 -m grpc_tools.protoc -I. --python_out=$(HOSPGEN) --grpc_python_out=$(HOSPGEN) federated.proto
