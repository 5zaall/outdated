import uuid

class Identifier:
    def __init__(self):
        pass

    def get_new_uuid(self):
        return uuid.uuid4()


if __name__=="__main__":
    u = Identifier()    

    print(u.get_new_uuid())
    