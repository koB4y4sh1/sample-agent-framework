from datetime import datetime


async def search(location, date:datetime):
    return  [{"outside":False,"event":"game"},{"outside":True,"event":"soccer"}]