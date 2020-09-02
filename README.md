### Build

```
$ docker-compose build
```

### Launch

```
$ docker-compose up
```

## rebuild .proto files using this command (if any changes were made to helloworld.proto)
```
$ python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. helloworld.proto
```
