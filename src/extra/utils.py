from typing import TypedDict


class UserProfile(TypedDict, total=False):
    username: str
    email: str
    age: int
    is_active: bool



user: UserProfile = {
    "username": "johndoe",
    "email": "mahedi.hrasel@gmail.cpm",
    "is_active": True,
    "age": 30,
}
print(user)


def get_user_email(user: UserProfile)  -> str | None:
    username = user.get("username")
    return username
