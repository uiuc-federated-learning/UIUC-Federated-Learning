### Run the coordinator locally

```
cd coordinator
python coordinator.py --ports 8001 8002
```

### Run a hospital instance

```
cd hospital
python hospital.py --port=<PORT_NUMBER>
```

### Rebuild generated files using this command 
(if any changes were made to federated.proto)
```
make proto
```
