from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os

def generate_aes_key(password, salt=b'salt_123'):
    password = password.encode()
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,  # 256-bit key for AES-256
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    key = kdf.derive(password)
    return key

def unpad(data):
    unpadder = padding.PKCS7(128).unpadder()
    unpadded_data = unpadder.update(data)
    try:
        return unpadded_data + unpadder.finalize()
    except ValueError:
        return data  # Return the data as is if there's an issue with padding

def decrypt_file(encrypted_file_path, key):
    with open(encrypted_file_path, 'rb') as infile:
        iv = infile.read(16)  # Read the initialization vector (IV)
        encrypted_data = infile.read()

    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted_data = decryptor.update(encrypted_data) + decryptor.finalize()

    unpadded_data = unpad(decrypted_data)

    decrypted_file_path = encrypted_file_path[:-4]  # Remove the .enc extension
    with open(decrypted_file_path, 'wb') as outfile:
        outfile.write(unpadded_data)

if __name__ == "__main__":
    folder_location = 'dot'  # Replace with your dataset folder path
    password = "MySecretPassword123"

    generated_key = generate_aes_key(password)
    print("Generated AES Key:", generated_key.hex())

    for root, _, files in os.walk(folder_location):
        for file in files:
            if file.endswith('.enc'):
                file_path = os.path.join(root, file)
                decrypt_file(file_path, generated_key)
