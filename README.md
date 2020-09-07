### Build

```
$ docker-compose build
```

### Launch

```
$ docker-compose up
```

### Rebuild .proto files using this command 
(if any changes were made to federated.proto)
```
$ python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. federated.proto
```
