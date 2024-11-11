from pydantic import BaseModel

class ParkingLotData(BaseModel):
    left: list = []
    middle: list = []
    right: list = []
