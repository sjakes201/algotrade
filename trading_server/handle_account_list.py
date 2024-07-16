def handle_account_list(data):
    for item in data:
        print(f"Name: {item['name']}")
        print(f"Account Type: {item['accountType']}")
        print(f"Active: {'Yes' if item['active'] else 'Inactive'}")
        print(f"ID: {item['id']}")
        print(f"UserID: {item['userId']}")
        print(f"Legal Status: {item['legalStatus']}")
        print(f"Margin Account Type: {item['marginAccountType']}")
        print(f"Risk Category: {item['riskCategoryId']}")
        print(f"Auto Liq Profile ID: {item['autoLiqProfileId']}")
        print(f"Clearing House ID: {item['clearingHouseId']}")
        print(f"Archived: {item['archived']}")
        print("")
