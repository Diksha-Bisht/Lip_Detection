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

def pad(data):
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(data)
    padded_data += padder.finalize()
    return padded_data

def encrypt_file(file_path, key):
    iv = b'1234567890123456'  # IV should be 16 bytes for AES

    with open(file_path, 'rb') as infile:
        data = infile.read()

    padded_data = pad(data)

    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

    encrypted_file_path = file_path + '.enc'
    with open(encrypted_file_path, 'wb') as outfile:
        outfile.write(iv)
        outfile.write(encrypted_data)

    os.remove(file_path)

if __name__ == "__main__":
    folder_location = 'Lip_Detection/dot'  # Replace with your dataset folder path
    password = "MySecretPassword123"

    generated_key = generate_aes_key(password)
    print("Generated AES Key:", generated_key.hex())

    for root, dirs, files in os.walk(folder_location):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                encrypt_file(file_path, generated_key)
