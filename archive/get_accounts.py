from tinkoff.invest import Client
import os

with Client(os.environ["TINKOFF_TOKEN"]) as client:
    accounts = client.users.get_accounts()
    for acc in accounts.accounts:
        print("Name:", acc.name)
        print("Type:", acc.type)
        print("ID:", acc.id)
        print()
