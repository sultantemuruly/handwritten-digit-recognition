from pydantic import BaseModel


class DigitImage(BaseModel):
    image: bytes
