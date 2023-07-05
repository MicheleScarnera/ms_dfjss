import dfjss_objects as dfjss

warehouse = dfjss.Warehouse()

for _ in range(15):
    print(warehouse.add_machine().features)

for _ in range(5):
    job = warehouse.add_job()
    print(job.features)
    for op in job.operations:
        print(f"\t{op.features}")