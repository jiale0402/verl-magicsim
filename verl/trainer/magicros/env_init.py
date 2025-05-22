import requests
import base64
import pickle as pkl

def start_magicros_env_single(server_url: str):
    def decode_data(encoded_data):
        decoded_data = base64.b64decode(encoded_data.encode("utf-8"))
        data = pkl.loads(decoded_data)
        return data
    resp = requests.post(f"{server_url}/reset")
    data = resp.json()["data"]
    obs, _ = decode_data(data)
    print(f"Connected to MagicROS server at {server_url}")
    return obs